#ifndef DREIDEL_TRANSFORMS_AUDIO_ENCODER_HPP
#define DREIDEL_TRANSFORMS_AUDIO_ENCODER_HPP

#include "../core/Tensor.hpp"
#include "../algo/FFT.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dreidel {
namespace transforms {

struct AudioEncoderConfig {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 80;
};

class AudioEncoder {
public:
    AudioEncoder(const AudioEncoderConfig& config = AudioEncoderConfig()) : config_(config) {
        init_window();
        init_mel_filters();
    }

    // Input: 1D audio samples (normalized -1 to 1). Assumes single batch for simplicity of API,
    // or flat vector. We can wrap in Tensor processing.
    // Returns: Tensor of shape (1, n_mels, n_frames)
    // Supports batch processing if input is provided as (Batch, Time)
    template <typename T, BackendType B>
    Tensor<T, B> forward(const Tensor<T, B>& input) {
        // Assume input is (Batch, Samples) or (Samples,)
        size_t batch_size = 1;
        size_t num_samples = input.size();

        if (input.shape().size() == 2) {
            batch_size = input.shape()[0];
            num_samples = input.shape()[1];
        }

        // Output dimensions
        // Number of frames: (num_samples) / hop_length + 1 (approx, depends on padding)
        // We use centered padding similar to librosa
        size_t n_frames = (num_samples) / config_.hop_length + 1; // Simplified

        Tensor<T, B> output({batch_size, (size_t)config_.n_mels, n_frames});
        output.fill(0);

        const T* in_data = input.data();
        T* out_data = output.data();

        // Process each item in batch
        for (size_t b = 0; b < batch_size; ++b) {
             const T* batch_in = in_data + b * num_samples;
             T* batch_out = out_data + b * config_.n_mels * n_frames;

             process_sample(batch_in, num_samples, batch_out, n_frames);
        }

        return output;
    }

private:
    AudioEncoderConfig config_;
    std::vector<float> window_;
    std::vector<std::vector<float>> mel_filters_; // (n_mels, n_fft/2 + 1)

    void init_window() {
        window_.resize(config_.n_fft);
        for (int i = 0; i < config_.n_fft; ++i) {
            // Hann window
            window_[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (config_.n_fft)));
        }
    }

    float hz_to_mel(float freq) {
        return 2595.0f * std::log10(1.0f + freq / 700.0f);
    }

    float mel_to_hz(float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    void init_mel_filters() {
        int n_mels = config_.n_mels;
        int n_fft = config_.n_fft;
        int sr = config_.sample_rate;

        float fmin = 0.0f;
        float fmax = sr / 2.0f;

        float mel_min = hz_to_mel(fmin);
        float mel_max = hz_to_mel(fmax);

        std::vector<float> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; ++i) {
            mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
        }

        int num_freqs = n_fft / 2 + 1;
        mel_filters_.resize(n_mels, std::vector<float>(num_freqs, 0.0f));

        for (int i = 0; i < n_mels; ++i) {
            float f_left = mel_points[i];
            float f_center = mel_points[i+1];
            float f_right = mel_points[i+2];

            for (int k = 0; k < num_freqs; ++k) {
                float freq = (float)k * sr / n_fft;

                if (freq >= f_left && freq <= f_center) {
                    mel_filters_[i][k] = (freq - f_left) / (f_center - f_left);
                } else if (freq > f_center && freq <= f_right) {
                    mel_filters_[i][k] = (f_right - freq) / (f_right - f_center);
                }
            }
        }
    }

    template <typename T>
    void process_sample(const T* input, size_t num_samples, T* output, size_t n_frames) {
        // Pad input (reflect or zero). Using Reflect padding of n_fft / 2
        int pad = config_.n_fft / 2;
        std::vector<float> padded_input(num_samples + 2 * pad);

        // Center padding: reflect
        for (int i = 0; i < pad; ++i) padded_input[i] = input[pad - i];
        for (size_t i = 0; i < num_samples; ++i) padded_input[pad + i] = input[i];
        for (int i = 0; i < pad; ++i) padded_input[pad + num_samples + i] = input[num_samples - 2 - i];

        // STFT loop
        std::vector<algo::FFT::Complex> fft_buffer;

        for (size_t t = 0; t < n_frames; ++t) {
            size_t start = t * config_.hop_length;
            if (start + config_.n_fft > padded_input.size()) break;

            // Windowing
            std::vector<float> frame(config_.n_fft);
            for (int i = 0; i < config_.n_fft; ++i) {
                frame[i] = padded_input[start + i] * window_[i];
            }

            // FFT
            algo::FFT::rfft(frame, fft_buffer);

            // Compute Power Spectrum
            std::vector<float> power_spec(fft_buffer.size());
            for (size_t i = 0; i < fft_buffer.size(); ++i) {
                float mag = std::abs(fft_buffer[i]);
                power_spec[i] = mag * mag;
            }

            // Apply Mel Filters
            for (int m = 0; m < config_.n_mels; ++m) {
                float mel_val = 0.0f;
                for (size_t k = 0; k < power_spec.size(); ++k) {
                    mel_val += power_spec[k] * mel_filters_[m][k];
                }

                // Log10
                mel_val = std::log10(std::max(mel_val, 1e-10f));

                // Store in output (Channel-first: output[m][t])
                // Or rather output[m * n_frames + t]
                output[m * n_frames + t] = static_cast<T>(mel_val);
            }
        }
    }
};

} // namespace transforms
} // namespace dreidel

#endif // DREIDEL_TRANSFORMS_AUDIO_ENCODER_HPP
