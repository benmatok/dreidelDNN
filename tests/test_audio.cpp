#include "../include/dreidel/transforms/AudioEncoder.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace dreidel;

void test_fft() {
    std::cout << "Testing FFT..." << std::endl;
    std::vector<algo::FFT::Complex> data(8);
    for (int i=0; i<8; ++i) data[i] = algo::FFT::Complex(i, 0);

    algo::FFT::fft(data);

    // Check DC component (sum)
    // Sum 0..7 = 28
    if (std::abs(data[0].real() - 28.0) > 1e-5) {
        std::cerr << "FFT DC component incorrect: " << data[0] << std::endl;
        exit(1);
    }
    std::cout << "FFT Passed." << std::endl;
}

void test_audio_encoder() {
    std::cout << "Testing AudioEncoder..." << std::endl;

    // Generate sine wave 440Hz
    int sr = 16000;
    float freq = 440.0f;
    size_t duration = sr; // 1 second

    std::vector<float> sine_wave(duration);
    for (size_t i=0; i<duration; ++i) {
        sine_wave[i] = std::sin(2.0f * M_PI * freq * i / sr);
    }

    Tensor<float> input({1, duration});
    std::copy(sine_wave.begin(), sine_wave.end(), input.data());

    transforms::AudioEncoderConfig config;
    config.n_mels = 80;
    transforms::AudioEncoder encoder(config);

    auto output = encoder.forward(input);

    std::cout << "Output shape: ";
    for (auto d : output.shape()) std::cout << d << " ";
    std::cout << std::endl;

    // Check if output is non-zero
    float max_val = -1e9;
    for (size_t i=0; i<output.size(); ++i) {
        if (output.data()[i] > max_val) max_val = output.data()[i];
    }

    std::cout << "Max Mel Value: " << max_val << std::endl;

    if (max_val <= -10.0) { // Log of very small number is negative
        std::cerr << "AudioEncoder output seems silent/empty." << std::endl;
        exit(1);
    }
    std::cout << "AudioEncoder Passed." << std::endl;
}

int main() {
    test_fft();
    test_audio_encoder();
    return 0;
}
