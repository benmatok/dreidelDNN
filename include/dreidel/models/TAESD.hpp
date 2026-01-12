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

    // Helper to zero out data
    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

struct ConvWeights {
    std::vector<float> w; // Flattened [In, H, W, Out] - Transposed!
    std::vector<float> b; // [Out]
    int in_c, out_c, k;
};

// Optimized 3x3 Conv with Padding=1, Stride=1
// Weights are expected to be in [In, K, K, Out] format (or [IC, KH, KW, OC])
inline void conv2d_3x3_avx2(const Tensor& input, Tensor& output, const ConvWeights& weights) {
    int H = input.h;
    int W = input.w;
    int IC = weights.in_c;
    int OC = weights.out_c;

    // Check dimensions
    if (input.c != IC) throw std::runtime_error("Input channels mismatch weights");
    if (output.c != OC) throw std::runtime_error("Output channels mismatch weights");
    if (output.h != H || output.w != W) throw std::runtime_error("Output resolution mismatch (only stride 1 supported)");

    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float* out_ptr = &output.data[(y * W + x) * OC];

            // 1. Initialize with Bias
            for (int o = 0; o < OC; o += 8) {
                if (o + 7 < OC) {
                     _mm256_storeu_ps(&out_ptr[o], _mm256_loadu_ps(&weights.b[o]));
                } else {
                    // Handle edge case if OC is not multiple of 8 (though TAESD usually uses 64/3 which is tricky for 3)
                    // For OC=3 (final layer), we can't use AVX store of 8 floats.
                    // Fallback for remainder
                    for (int rem = o; rem < OC; ++rem) {
                        out_ptr[rem] = weights.b[rem];
                    }
                }
            }

            // 2. Convolution Accumulation
            for (int ky = -1; ky <= 1; ++ky) {
                int in_y = y + ky;
                if (in_y < 0 || in_y >= H) continue; // Zero padding check

                for (int kx = -1; kx <= 1; ++kx) {
                    int in_x = x + kx;
                    if (in_x < 0 || in_x >= W) continue;

                    const float* in_ptr = &input.data[(in_y * W + in_x) * IC];

                    // Inner Loop: Input Channels -> Output Channels
                    for (int ic = 0; ic < IC; ++ic) {
                        float pixel_val = in_ptr[ic];
                        __m256 v_pixel = _mm256_set1_ps(pixel_val);

                        // Weight Index: [IC, KH, KW, OC] flattened
                        // KH=3, KW=3.
                        // ky is -1..1, mapped to 0..2 -> ky+1
                        // kx is -1..1, mapped to 0..2 -> kx+1
                        // Index = ic * (3*3*OC) + (ky+1) * (3*OC) + (kx+1) * OC
                        int w_idx_base = (ic * 9 + (ky + 1) * 3 + (kx + 1)) * OC;

                        for (int o = 0; o < OC; o += 8) {
                            if (o + 7 < OC) {
                                __m256 v_w = _mm256_loadu_ps(&weights.w[w_idx_base + o]);
                                __m256 v_acc = _mm256_loadu_ps(&out_ptr[o]);
                                _mm256_storeu_ps(&out_ptr[o], _mm256_fmadd_ps(v_pixel, v_w, v_acc));
                            } else {
                                // Scalar fallback
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

class Decoder {
    std::vector<ConvWeights> layers;

public:
    void load_from_file(const char* filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open model file");
        }

        // Define architecture to know what to read
        // TAESD Decoder standard (4 -> 3)
        // 1. Conv(4->64)
        // 2. Block(64): Conv(64->64), Conv(64->64), Conv(64->64)
        // ...

        // Sequence of channels for all Conv2d layers in order
        struct LayerDef { int in, out; };
        std::vector<LayerDef> defs;

        // 1. Start: conv(4, 64)
        defs.push_back({4, 64});

        // 2. 3x Block(64)
        for(int i=0; i<3; ++i) { // 3 Blocks
            defs.push_back({64, 64}); // conv1
            defs.push_back({64, 64}); // conv2
            defs.push_back({64, 64}); // conv3
        }

        // 3. Up(2), Conv(64, 64)
        defs.push_back({64, 64});

        // 4. 3x Block(64)
        for(int i=0; i<3; ++i) {
            defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64});
        }

        // 5. Up(2), Conv(64, 64)
        defs.push_back({64, 64});

        // 6. 3x Block(64)
        for(int i=0; i<3; ++i) {
            defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64});
        }

        // 7. Up(2), Conv(64, 64)
        defs.push_back({64, 64});

        // 8. 1x Block(64)
        defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64});

        // 9. Final Conv(64, 3)
        defs.push_back({64, 3});

        // Read layers
        for (const auto& d : defs) {
            ConvWeights cw;
            cw.in_c = d.in;
            cw.out_c = d.out;
            cw.k = 3;

            // Weights: In * Out * K * K floats
            int w_count = cw.in_c * cw.out_c * 9;
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

        // Check if extra data exists? Optional.
        std::cout << "Loaded TAESD Decoder with " << layers.size() << " layers." << std::endl;
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
        // We can write back to x if dimensions match (they do 64->64)
        conv2d_3x3_avx2(t2, x, layers[start_layer_idx + 2]);

        // Skip
        add_inplace(x, x_orig);

        // Fuse
        relu(x);

        return start_layer_idx + 3;
    }

    void forward(Tensor& latent, Tensor& image) {
        // Latent input: usually 64x64x4 (for 512x512 image) or similar
        // TAESD usually takes small latent.

        // 0. Clamp (Optional but in TAESD)
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
        // Note: TAESD structure: Upsample -> Conv(bias=False) -> (Blocks starts with Conv)
        // The conv after upsample is just a conv. It doesn't have ReLU after it in Decoder definition?
        // Decoder def: ... nn.Upsample(), conv(64, 64), Block ...
        // `conv` helper has padding=1.
        // `conv` is just Conv2d. No ReLU in `conv` helper?
        // Helper: def conv(...): return nn.Conv2d(...)
        // So NO ReLU after the Upsample Conv.

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
        // Ensure image tensor is correct size
        if (image.h != f4_conv.h || image.w != f4_conv.w || image.c != 3) {
             // Resize or throw? Assuming caller allocated correctly or we resize
             image = Tensor(f4_conv.h, f4_conv.w, 3);
        }
        conv2d_3x3_avx2(f4_conv, image, layers[l_idx++]);
        // No activation at end
    }
};

} // namespace taesd
} // namespace dreidel

#endif
