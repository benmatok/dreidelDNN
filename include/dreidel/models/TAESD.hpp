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
    std::vector<float> w; // Flattened [In, H, W, Out]
    std::vector<float> b; // [Out]
    int in_c, out_c, k;
};

// --- AVX2 Kernels ---

// Optimized 3x3 Conv with Padding=1, Stride=1
// Uses 4-pixel blocking to reduce weight bandwidth
inline void conv2d_3x3_avx2(const Tensor& input, Tensor& output, const ConvWeights& weights) {
    int H = input.h;
    int W = input.w;
    int IC = weights.in_c;
    int OC = weights.out_c;

    // Output pointers
    float* out_base = output.data.data();
    const float* in_base = input.data.data();
    const float* w_base = weights.w.data();
    const float* b_base = weights.b.data();

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < H; ++y) {
        // Process 4 pixels at a time (x, x+1, x+2, x+3)
        for (int x = 0; x < W; x += 4) {
            // Handle boundary
            int valid_pixels = std::min(4, W - x);

            // Registers for Accumulators: 4 pixels * 16 channels (2 regs)
            // We loop OC in chunks of 16.
            // If OC < 16, we handle tail.

            for (int oc = 0; oc < OC; oc += 16) {
                int oc_rem = std::min(16, OC - oc);

                // Accumulators (2 regs per pixel, 4 pixels)
                // P0_0, P0_1 (ch 0-7, 8-15)
                // P1_0, P1_1 ...
                __m256 acc[4][2];

                // Initialize with Bias
                for (int p = 0; p < valid_pixels; ++p) {
                    if (oc_rem >= 8)  acc[p][0] = _mm256_loadu_ps(&b_base[oc]);
                    else {
                        // partial load
                        float tmp[8]={0}; for(int k=0;k<oc_rem;++k) tmp[k]=b_base[oc+k];
                        acc[p][0] = _mm256_loadu_ps(tmp);
                    }
                    if (oc_rem > 8) {
                        if (oc_rem >= 16) acc[p][1] = _mm256_loadu_ps(&b_base[oc+8]);
                        else {
                            float tmp[8]={0}; for(int k=0;k<oc_rem-8;++k) tmp[k]=b_base[oc+8+k];
                            acc[p][1] = _mm256_loadu_ps(tmp);
                        }
                    } else {
                        acc[p][1] = _mm256_setzero_ps();
                    }
                }

                // Kernel Loop
                for (int ky = -1; ky <= 1; ++ky) {
                    int in_y = y + ky;
                    bool y_valid = (in_y >= 0 && in_y < H);

                    for (int kx = -1; kx <= 1; ++kx) {
                        // To optimize, we loop IC.
                        for (int ic = 0; ic < IC; ++ic) {
                            // Load Weights for this OC chunk (16 floats, 2 regs)
                            // W is [IC, 3, 3, OC]
                            // Index: (ic * 9 + (ky+1)*3 + (kx+1)) * OC + oc
                            int w_idx = (ic * 9 + (ky + 1) * 3 + (kx + 1)) * OC + oc;

                            __m256 w0, w1;
                            if (oc_rem >= 8) w0 = _mm256_loadu_ps(&w_base[w_idx]);
                            if (oc_rem > 8)  w1 = _mm256_loadu_ps(&w_base[w_idx+8]);

                            // Iterate pixels
                            for (int p = 0; p < valid_pixels; ++p) {
                                if (!y_valid) continue; // Padding Y
                                int in_x = x + p + kx;
                                if (in_x < 0 || in_x >= W) continue; // Padding X

                                // Load pixel value (broadcast)
                                float val = in_base[(in_y * W + in_x) * IC + ic];
                                __m256 v_val = _mm256_set1_ps(val);

                                // FMADD
                                if (oc_rem >= 8) acc[p][0] = _mm256_fmadd_ps(v_val, w0, acc[p][0]);
                                if (oc_rem > 8)  acc[p][1] = _mm256_fmadd_ps(v_val, w1, acc[p][1]);
                            }
                        }
                    }
                }

                // Store
                for (int p = 0; p < valid_pixels; ++p) {
                    float* dst = &out_base[((y * W) + x + p) * OC + oc];
                    if (oc_rem >= 8) _mm256_storeu_ps(dst, acc[p][0]);
                    else {
                        float tmp[8]; _mm256_storeu_ps(tmp, acc[p][0]);
                        for(int k=0; k<oc_rem; ++k) dst[k] = tmp[k];
                    }

                    if (oc_rem > 8) {
                        if (oc_rem >= 16) _mm256_storeu_ps(dst+8, acc[p][1]);
                        else {
                            float tmp[8]; _mm256_storeu_ps(tmp, acc[p][1]);
                            for(int k=0; k<oc_rem-8; ++k) dst[8+k] = tmp[k];
                        }
                    }
                }
            }
        }
    }
}

// Stride 2 Version (Simplified Blocking)
inline void conv2d_3x3_s2_avx2(const Tensor& input, Tensor& output, const ConvWeights& weights) {
    int H_in = input.h;
    int W_in = input.w;
    int H_out = output.h;
    int W_out = output.w;
    int IC = weights.in_c;
    int OC = weights.out_c;

    const float* in_base = input.data.data();
    float* out_base = output.data.data();
    const float* w_base = weights.w.data();
    const float* b_base = weights.b.data();

    #pragma omp parallel for schedule(dynamic)
    for (int y_out = 0; y_out < H_out; ++y_out) {
        int y_in_center = y_out * 2;

        for (int x_out = 0; x_out < W_out; x_out += 4) {
            int valid_pixels = std::min(4, W_out - x_out);

            for (int oc = 0; oc < OC; oc += 16) {
                int oc_rem = std::min(16, OC - oc);
                __m256 acc[4][2];

                // Init
                for (int p = 0; p < valid_pixels; ++p) {
                    if (oc_rem >= 8)  acc[p][0] = _mm256_loadu_ps(&b_base[oc]);
                    else { float tmp[8]={0}; for(int k=0;k<oc_rem;++k) tmp[k]=b_base[oc+k]; acc[p][0] = _mm256_loadu_ps(tmp); }
                    if (oc_rem > 8) {
                         if (oc_rem >= 16) acc[p][1] = _mm256_loadu_ps(&b_base[oc+8]);
                         else { float tmp[8]={0}; for(int k=0;k<oc_rem-8;++k) tmp[k]=b_base[oc+8+k]; acc[p][1] = _mm256_loadu_ps(tmp); }
                    } else acc[p][1] = _mm256_setzero_ps();
                }

                for (int ky = -1; ky <= 1; ++ky) {
                    int in_y = y_in_center + ky;
                    bool y_valid = (in_y >= 0 && in_y < H_in);

                    for (int kx = -1; kx <= 1; ++kx) {
                         for (int ic = 0; ic < IC; ++ic) {
                            int w_idx = (ic * 9 + (ky + 1) * 3 + (kx + 1)) * OC + oc;
                            __m256 w0, w1;
                            if (oc_rem >= 8) w0 = _mm256_loadu_ps(&w_base[w_idx]);
                            if (oc_rem > 8)  w1 = _mm256_loadu_ps(&w_base[w_idx+8]);

                            for (int p = 0; p < valid_pixels; ++p) {
                                if (!y_valid) continue;
                                int in_x = (x_out + p) * 2 + kx; // Stride 2 logic
                                if (in_x < 0 || in_x >= W_in) continue;

                                float val = in_base[(in_y * W_in + in_x) * IC + ic];
                                __m256 v_val = _mm256_set1_ps(val);

                                if (oc_rem >= 8) acc[p][0] = _mm256_fmadd_ps(v_val, w0, acc[p][0]);
                                if (oc_rem > 8)  acc[p][1] = _mm256_fmadd_ps(v_val, w1, acc[p][1]);
                            }
                         }
                    }
                }

                // Store
                for (int p = 0; p < valid_pixels; ++p) {
                    float* dst = &out_base[((y_out * W_out) + x_out + p) * OC + oc];
                    if (oc_rem >= 8) _mm256_storeu_ps(dst, acc[p][0]);
                    else { float tmp[8]; _mm256_storeu_ps(tmp, acc[p][0]); for(int k=0;k<oc_rem;++k) dst[k]=tmp[k]; }
                    if (oc_rem > 8) {
                        if (oc_rem >= 16) _mm256_storeu_ps(dst+8, acc[p][1]);
                        else { float tmp[8]; _mm256_storeu_ps(tmp, acc[p][1]); for(int k=0;k<oc_rem-8;++k) dst[8+k]=tmp[k]; }
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
     int size = t.h * t.w * t.c;
     #pragma omp parallel for
     for (int i = 0; i < size; ++i) {
         t.data[i] = std::tanh(t.data[i] / 3.0f) * 3.0f;
     }
}

inline void add_inplace(Tensor& out, const Tensor& in) {
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

inline void upsample_nearest(const Tensor& in, Tensor& out) {
    #pragma omp parallel for
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            int src_x = x / 2;
            int src_y = y / 2;
            const float* src = &in.data[(src_y * in.w + src_x) * in.c];
            float* dst = &out.data[(y * out.w + x) * out.c];
            for(int c=0; c < in.c; ++c) dst[c] = src[c];
        }
    }
}

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

            size_t w_count = cw.in_c * cw.out_c * 9;
            cw.w.resize(w_count);
            f.read(reinterpret_cast<char*>(cw.w.data()), w_count * sizeof(float));

            cw.b.resize(cw.out_c);
            f.read(reinterpret_cast<char*>(cw.b.data()), cw.out_c * sizeof(float));

            layers.push_back(std::move(cw));
        }
    }

    int run_block(Tensor& x, int start_layer_idx) {
        Tensor x_orig = x;

        Tensor t1(x.h, x.w, 64);
        conv2d_3x3_avx2(x, t1, layers[start_layer_idx]);
        relu(t1);

        Tensor t2(x.h, x.w, 64);
        conv2d_3x3_avx2(t1, t2, layers[start_layer_idx + 1]);
        relu(t2);

        conv2d_3x3_avx2(t2, x, layers[start_layer_idx + 2]);

        add_inplace(x, x_orig);
        relu(x);
        return start_layer_idx + 3;
    }
};

class Decoder : public ModelBase {
public:
    void load_from_file(const char* filename) {
        std::vector<std::pair<int, int>> defs;
        defs.push_back({4, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64});
        defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64});
        defs.push_back({64, 3});
        load_layers(filename, defs);
    }

    void forward(Tensor& latent, Tensor& image) {
        clamp_tanh_3(latent);
        int l_idx = 0;
        Tensor f1(latent.h, latent.w, 64);
        conv2d_3x3_avx2(latent, f1, layers[l_idx++]);
        relu(f1);
        for(int i=0; i<3; ++i) l_idx = run_block(f1, l_idx);
        Tensor f2(f1.h * 2, f1.w * 2, 64);
        upsample_nearest(f1, f2);
        Tensor f2_conv(f2.h, f2.w, 64);
        conv2d_3x3_avx2(f2, f2_conv, layers[l_idx++]);
        for(int i=0; i<3; ++i) l_idx = run_block(f2_conv, l_idx);
        Tensor f3(f2_conv.h * 2, f2_conv.w * 2, 64);
        upsample_nearest(f2_conv, f3);
        Tensor f3_conv(f3.h, f3.w, 64);
        conv2d_3x3_avx2(f3, f3_conv, layers[l_idx++]);
        for(int i=0; i<3; ++i) l_idx = run_block(f3_conv, l_idx);
        Tensor f4(f3_conv.h * 2, f3_conv.w * 2, 64);
        upsample_nearest(f3_conv, f4);
        Tensor f4_conv(f4.h, f4.w, 64);
        conv2d_3x3_avx2(f4, f4_conv, layers[l_idx++]);
        l_idx = run_block(f4_conv, l_idx);
        if (image.h != f4_conv.h || image.w != f4_conv.w || image.c != 3) image = Tensor(f4_conv.h, f4_conv.w, 3);
        conv2d_3x3_avx2(f4_conv, image, layers[l_idx++]);
    }
};

class Encoder : public ModelBase {
public:
    void load_from_file(const char* filename) {
        std::vector<std::pair<int, int>> defs;
        defs.push_back({3, 64});
        defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64});
        defs.push_back({64, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 64});
        for(int i=0; i<3; ++i) { defs.push_back({64, 64}); defs.push_back({64, 64}); defs.push_back({64, 64}); }
        defs.push_back({64, 4});
        load_layers(filename, defs);
    }

    void forward(Tensor& image, Tensor& latent) {
        int l_idx = 0;
        Tensor f1(image.h, image.w, 64);
        conv2d_3x3_avx2(image, f1, layers[l_idx++]);
        l_idx = run_block(f1, l_idx);
        Tensor f2((f1.h-1)/2+1, (f1.w-1)/2+1, 64);
        conv2d_3x3_s2_avx2(f1, f2, layers[l_idx++]);
        for(int i=0; i<3; ++i) l_idx = run_block(f2, l_idx);
        Tensor f3((f2.h-1)/2+1, (f2.w-1)/2+1, 64);
        conv2d_3x3_s2_avx2(f2, f3, layers[l_idx++]);
        for(int i=0; i<3; ++i) l_idx = run_block(f3, l_idx);
        Tensor f4((f3.h-1)/2+1, (f3.w-1)/2+1, 64);
        conv2d_3x3_s2_avx2(f3, f4, layers[l_idx++]);
        for(int i=0; i<3; ++i) l_idx = run_block(f4, l_idx);
        if (latent.h != f4.h || latent.w != f4.w || latent.c != 4) latent = Tensor(f4.h, f4.w, 4);
        conv2d_3x3_avx2(f4, latent, layers[l_idx++]);
    }
};

} // namespace taesd
} // namespace dreidel

#endif
