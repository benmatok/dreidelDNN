#ifndef DREIDEL_MODELS_TAESD_HPP
#define DREIDEL_MODELS_TAESD_HPP

#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <immintrin.h> // AVX2
#include <omp.h>       // OpenMP
#include <algorithm>   // std::copy_n, std::max
#include <stdexcept>

namespace dreidel {
namespace taesd {

// Lightweight Tensor struct as requested
struct Tensor {
    int h, w, c;
    std::vector<float> data;

    Tensor(int _h, int _w, int _c) : h(_h), w(_w), c(_c) {
        data.resize(h * w * c);
    }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }

    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

struct ConvWeights {
    std::vector<float> w; // Flattened [In, H, W, Out] - Transposed!
    std::vector<float> b; // [Out]
    int in_c, out_c, k;
};

// --- AVX2 Kernels ---

// Optimized 3x3 Conv with Padding=1, Stride=1
inline void conv2d_3x3_avx2(const Tensor& input, Tensor& output, const ConvWeights& weights) {
    int H = input.h;
    int W = input.w;
    int IC = weights.in_c;
    int OC = weights.out_c;

    if (input.c != IC) throw std::runtime_error("Input channels mismatch weights");
    if (output.c != OC) throw std::runtime_error("Output channels mismatch weights");
    // Verify output spatial dims
    if (output.h != H || output.w != W) throw std::runtime_error("Output resolution mismatch (expected stride 1)");

    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float* out_ptr = &output.data[(y * W + x) * OC];

            // 1. Initialize with Bias
            for (int o = 0; o < OC; o += 8) {
                if (o + 7 < OC) {
                     _mm256_storeu_ps(&out_ptr[o], _mm256_loadu_ps(&weights.b[o]));
                } else {
                    for (int rem = o; rem < OC; ++rem) out_ptr[rem] = weights.b[rem];
                }
            }

            // 2. Convolution Accumulation
            for (int ky = -1; ky <= 1; ++ky) {
                int in_y = y + ky;
                if (in_y < 0 || in_y >= H) continue;

                for (int kx = -1; kx <= 1; ++kx) {
                    int in_x = x + kx;
                    if (in_x < 0 || in_x >= W) continue;

                    const float* in_ptr = &input.data[(in_y * W + in_x) * IC];

                    for (int ic = 0; ic < IC; ++ic) {
                        float pixel_val = in_ptr[ic];
                        __m256 v_pixel = _mm256_set1_ps(pixel_val);

                        // Weight Index: [IC, KH, KW, OC]
                        int w_idx_base = (ic * 9 + (ky + 1) * 3 + (kx + 1)) * OC;

                        for (int o = 0; o < OC; o += 8) {
                            if (o + 7 < OC) {
                                __m256 v_w = _mm256_loadu_ps(&weights.w[w_idx_base + o]);
                                __m256 v_acc = _mm256_loadu_ps(&out_ptr[o]);
                                _mm256_storeu_ps(&out_ptr[o], _mm256_fmadd_ps(v_pixel, v_w, v_acc));
                            } else {
                                for (int rem = o; rem < OC; ++rem) {
                                    out_ptr[rem] += pixel_val * weights.w[w_idx_base + rem];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Optimized 3x3 Conv with Padding=1, Stride=2
inline void conv2d_3x3_s2_avx2(const Tensor& input, Tensor& output, const ConvWeights& weights) {
    int H_in = input.h;
    int W_in = input.w;
    int H_out = output.h;
    int W_out = output.w;
    int IC = weights.in_c;
    int OC = weights.out_c;

    // Check basic dimensions
    if (input.c != IC) throw std::runtime_error("Input channels mismatch weights");
    if (output.c != OC) throw std::runtime_error("Output channels mismatch weights");
    // Verify spatial: Output = floor((Input + 2*Pad - K) / Stride) + 1
    // Pad=1, K=3, Stride=2 => floor((Input + 2 - 3)/2) + 1 = floor((Input-1)/2) + 1
    // e.g., 64 -> floor(63/2)+1 = 31+1 = 32. Correct.
    int expected_h = (H_in - 1) / 2 + 1;
    int expected_w = (W_in - 1) / 2 + 1;
    if (H_out != expected_h || W_out != expected_w) throw std::runtime_error("Output resolution mismatch (expected stride 2)");

    #pragma omp parallel for
    for (int y_out = 0; y_out < H_out; ++y_out) {
        int y_in_center = y_out * 2; // Stride 2
        for (int x_out = 0; x_out < W_out; ++x_out) {
            int x_in_center = x_out * 2; // Stride 2

            float* out_ptr = &output.data[(y_out * W_out + x_out) * OC];

            // 1. Initialize with Bias
            for (int o = 0; o < OC; o += 8) {
                if (o + 7 < OC) {
                     _mm256_storeu_ps(&out_ptr[o], _mm256_loadu_ps(&weights.b[o]));
                } else {
                    for (int rem = o; rem < OC; ++rem) out_ptr[rem] = weights.b[rem];
                }
            }

            // 2. Convolution Accumulation
            for (int ky = -1; ky <= 1; ++ky) {
                int in_y = y_in_center + ky;
                if (in_y < 0 || in_y >= H_in) continue;

                for (int kx = -1; kx <= 1; ++kx) {
                    int in_x = x_in_center + kx;
                    if (in_x < 0 || in_x >= W_in) continue;

                    const float* in_ptr = &input.data[(in_y * W_in + in_x) * IC];

                    for (int ic = 0; ic < IC; ++ic) {
                        float pixel_val = in_ptr[ic];
                        __m256 v_pixel = _mm256_set1_ps(pixel_val);

                        // Weight Index: [IC, KH, KW, OC]
                        int w_idx_base = (ic * 9 + (ky + 1) * 3 + (kx + 1)) * OC;

                        for (int o = 0; o < OC; o += 8) {
                            if (o + 7 < OC) {
                                __m256 v_w = _mm256_loadu_ps(&weights.w[w_idx_base + o]);
                                __m256 v_acc = _mm256_loadu_ps(&out_ptr[o]);
                                _mm256_storeu_ps(&out_ptr[o], _mm256_fmadd_ps(v_pixel, v_w, v_acc));
                            } else {
                                for (int rem = o; rem < OC; ++rem) {
                                    out_ptr[rem] += pixel_val * weights.w[w_idx_base + rem];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


inline void relu(Tensor& t) {
    int size = t.h * t.w * t.c;
    __m256 zero = _mm256_setzero_ps();
    #pragma omp parallel for
    for (int i = 0; i < size; i += 8) {
        if (i + 7 < size) {
            __m256 v = _mm256_loadu_ps(&t.data[i]);
            _mm256_storeu_ps(&t.data[i], _mm256_max_ps(v, zero));
        } else {
            for (int j = i; j < size; ++j) {
                t.data[j] = std::max(0.0f, t.data[j]);
            }
        }
    }
}

inline void clamp_tanh_3(Tensor& t) {
     // Tanh(x / 3) * 3
     int size = t.h * t.w * t.c;
     #pragma omp parallel for
     for (int i = 0; i < size; ++i) {
         t.data[i] = std::tanh(t.data[i] / 3.0f) * 3.0f;
     }
}

inline void add_inplace(Tensor& out, const Tensor& in) {
    // out += in
    if (out.data.size() != in.data.size()) return;
    int size = out.data.size();
    #pragma omp parallel for
    for (int i = 0; i < size; i+=8) {
        if (i+7 < size) {
             _mm256_storeu_ps(&out.data[i], _mm256_add_ps(_mm256_loadu_ps(&out.data[i]), _mm256_loadu_ps(&in.data[i])));
        } else {
            for (int j=i; j<size; ++j) out.data[j] += in.data[j];
        }
    }
}

// Standard Nearest Neighbor
inline void upsample_nearest(const Tensor& in, Tensor& out) {
    // Assumes out is 2x in
    #pragma omp parallel for
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            int src_x = x / 2;
            int src_y = y / 2;
            // Copy pixel vector (C floats)
            const float* src = &in.data[(src_y * in.w + src_x) * in.c];
            float* dst = &out.data[(y * out.w + x) * out.c];
            for(int c=0; c < in.c; ++c) dst[c] = src[c];
        }
    }
}

// Common base for handling layers
class ModelBase {
protected:
    std::vector<ConvWeights> layers;

public:
    void load_layers(const char* filename, const std::vector<std::pair<int, int>>& defs) {
         std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open model file");
        }

        for (const auto& d : defs) {
            ConvWeights cw;
            cw.in_c = d.first;
            cw.out_c = d.second;
            cw.k = 3;

            // Weights: In * Out * K * K floats
            size_t w_count = cw.in_c * cw.out_c * 9;
            cw.w.resize(w_count);
            f.read(reinterpret_cast<char*>(cw.w.data()), w_count * sizeof(float));

            // Bias: Out floats
            cw.b.resize(cw.out_c);
            f.read(reinterpret_cast<char*>(cw.b.data()), cw.out_c * sizeof(float));

            layers.push_back(std::move(cw));
        }

        if (f.fail()) {
             throw std::runtime_error("Error reading model file (file too short?)");
        }
        std::cout << "Loaded " << layers.size() << " layers from " << filename << std::endl;
    }

    // Helper to run a Block
    // block_idx is global index in layers vector
    // returns next layer index
    int run_block(Tensor& x, int start_layer_idx) {
        // Block is:
        // x_orig = x
        // x = Conv(x) -> ReLU -> Conv(x) -> ReLU -> Conv(x)
        // x = x + x_orig
        // x = ReLU(x)

        Tensor x_orig = x; // Copy for skip connection

        // Conv 1
        Tensor t1(x.h, x.w, 64);
        conv2d_3x3_avx2(x, t1, layers[start_layer_idx]);
        relu(t1);

        // Conv 2
        Tensor t2(x.h, x.w, 64);
        conv2d_3x3_avx2(t1, t2, layers[start_layer_idx + 1]);
        relu(t2);

        // Conv 3
        conv2d_3x3_avx2(t2, x, layers[start_layer_idx + 2]);

        // Skip
        add_inplace(x, x_orig);

        // Fuse
        relu(x);

        return start_layer_idx + 3;
    }
};

class Decoder : public ModelBase {
public:
    void load_from_file(const char* filename) {
        // Decoder (4 -> 3)
        // 1. Conv(4->64)
        // 2. 3x Block(64)
        // 3. Up(2), Conv(64, 64)
        // 4. 3x Block(64)
        // 5. Up(2), Conv(64, 64)
        // 6. 3x Block(64)
        // 7. Up(2), Conv(64, 64)
        // 8. 1x Block(64)
        // 9. Conv(64, 3)

        std::vector<std::pair<int, int>> defs;
        defs.push_back({4, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64}); // Up
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64}); // Up
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64}); // Up
        defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); // 1x Block
        defs.push_back({64, 3});

        load_layers(filename, defs);
    }

    void forward(Tensor& latent, Tensor& image) {
        // 0. Clamp
        clamp_tanh_3(latent);

        int l_idx = 0;

        // 1. Initial Conv (4 -> 64)
        Tensor f1(latent.h, latent.w, 64);
        conv2d_3x3_avx2(latent, f1, layers[l_idx++]);
        relu(f1);

        // 2. 3x Blocks
        for(int i=0; i<3; ++i) l_idx = run_block(f1, l_idx);

        // 3. Up + Conv
        Tensor f2(f1.h * 2, f1.w * 2, 64);
        upsample_nearest(f1, f2);
        Tensor f2_conv(f2.h, f2.w, 64);
        conv2d_3x3_avx2(f2, f2_conv, layers[l_idx++]);

        // 4. 3x Blocks
        for(int i=0; i<3; ++i) l_idx = run_block(f2_conv, l_idx);

        // 5. Up + Conv
        Tensor f3(f2_conv.h * 2, f2_conv.w * 2, 64);
        upsample_nearest(f2_conv, f3);
        Tensor f3_conv(f3.h, f3.w, 64);
        conv2d_3x3_avx2(f3, f3_conv, layers[l_idx++]);

        // 6. 3x Blocks
        for(int i=0; i<3; ++i) l_idx = run_block(f3_conv, l_idx);

        // 7. Up + Conv
        Tensor f4(f3_conv.h * 2, f3_conv.w * 2, 64);
        upsample_nearest(f3_conv, f4);
        Tensor f4_conv(f4.h, f4.w, 64);
        conv2d_3x3_avx2(f4, f4_conv, layers[l_idx++]);

        // 8. 1x Block
        l_idx = run_block(f4_conv, l_idx);

        // 9. Final Conv (64 -> 3)
        if (image.h != f4_conv.h || image.w != f4_conv.w || image.c != 3) {
             image = Tensor(f4_conv.h, f4_conv.w, 3);
        }
        conv2d_3x3_avx2(f4_conv, image, layers[l_idx++]);
    }
};

class Encoder : public ModelBase {
public:
    void load_from_file(const char* filename) {
        // Encoder (3 -> 4)
        // 1. Conv(3, 64)
        // 2. Block(64)
        // 3. Conv(64, 64, stride=2, bias=False)
        // 4. 3x Block(64)
        // 5. Conv(64, 64, stride=2, bias=False)
        // 6. 3x Block(64)
        // 7. Conv(64, 64, stride=2, bias=False)
        // 8. 3x Block(64)
        // 9. Conv(64, 4)

        std::vector<std::pair<int, int>> defs;
        defs.push_back({3, 64});
        defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); // 1x Block
        defs.push_back({64, 64}); // Down
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64}); // Down
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64}); // Down
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 4});

        load_layers(filename, defs);
    }

    void forward(Tensor& image, Tensor& latent) {
        int l_idx = 0;

        // 1. Initial Conv
        Tensor f1(image.h, image.w, 64);
        conv2d_3x3_avx2(image, f1, layers[l_idx++]);
        // Note: Python Encoder code says: conv(3, 64), Block(64, 64).
        // Does initial conv have ReLU?
        // tools/taesd.py: `conv(3, 64)` -> nn.Conv2d(..., padding=1). No ReLU.
        // Block has internal ReLU.
        // So f1 is output of Conv.

        // 2. 1x Block
        l_idx = run_block(f1, l_idx);

        // 3. Down (Stride 2)
        Tensor f2((f1.h-1)/2+1, (f1.w-1)/2+1, 64);
        conv2d_3x3_s2_avx2(f1, f2, layers[l_idx++]);
        // Python: conv(64, 64, stride=2, bias=False). No ReLU in conv wrapper.

        // 4. 3x Blocks
        for(int i=0; i<3; ++i) l_idx = run_block(f2, l_idx);

        // 5. Down (Stride 2)
        Tensor f3((f2.h-1)/2+1, (f2.w-1)/2+1, 64);
        conv2d_3x3_s2_avx2(f2, f3, layers[l_idx++]);

        // 6. 3x Blocks
        for(int i=0; i<3; ++i) l_idx = run_block(f3, l_idx);

        // 7. Down (Stride 2)
        Tensor f4((f3.h-1)/2+1, (f3.w-1)/2+1, 64);
        conv2d_3x3_s2_avx2(f3, f4, layers[l_idx++]);

        // 8. 3x Blocks
        for(int i=0; i<3; ++i) l_idx = run_block(f4, l_idx);

        // 9. Final Conv (64 -> 4)
        if (latent.h != f4.h || latent.w != f4.w || latent.c != 4) {
            latent = Tensor(f4.h, f4.w, 4);
        }
        conv2d_3x3_avx2(f4, latent, layers[l_idx++]);
        // Python: conv(64, latent_channels). No ReLU.
    }
};

} // namespace taesd
} // namespace dreidel

#endif
