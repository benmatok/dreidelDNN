#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "../hal/ops.hpp"

namespace dreidel {
namespace kernels {

// --- AVX-512 Helper Wrappers ---
#ifdef __AVX512F__
using Vec512 = __m512;
inline Vec512 Load512(const float* p) { return _mm512_loadu_ps(p); }
inline void Store512(float* p, Vec512 v) { _mm512_storeu_ps(p, v); }
inline Vec512 Add512(Vec512 a, Vec512 b) { return _mm512_add_ps(a, b); }
inline Vec512 Sub512(Vec512 a, Vec512 b) { return _mm512_sub_ps(a, b); }
inline Vec512 Mul512(Vec512 a, Vec512 b) { return _mm512_mul_ps(a, b); }
#endif

// --- AVX2 Fallback Wrappers ---
#ifdef __AVX2__
using Vec256 = __m256;
inline Vec256 Load256(const float* p) { return _mm256_loadu_ps(p); }
inline void Store256(float* p, Vec256 v) { _mm256_storeu_ps(p, v); }
inline Vec256 Add256(Vec256 a, Vec256 b) { return _mm256_add_ps(a, b); }
inline Vec256 Sub256(Vec256 a, Vec256 b) { return _mm256_sub_ps(a, b); }
inline Vec256 Mul256(Vec256 a, Vec256 b) { return _mm256_mul_ps(a, b); }
#endif

// Butterfly Operation on Vectors
#ifdef __AVX512F__
inline void Butterfly512(Vec512& a, Vec512& b) {
    Vec512 sum = Add512(a, b);
    Vec512 sub = Sub512(a, b);
    a = sum;
    b = sub;
}
#endif

#ifdef __AVX2__
inline void Butterfly256(Vec256& a, Vec256& b) {
    Vec256 sum = Add256(a, b);
    Vec256 sub = Sub256(a, b);
    a = sum;
    b = sub;
}
#endif

// 1D FWHT of size 64
// Input: Pointer to start of data (NHWC layout implies this processes SPATIAL dimension?)
// Wait, "Horizontal Spectral Mix (Row-wise)".
// If NHWC: [N, H, W, C]. Row is [W].
// Elements of a row are separated by C.
// The plan says "FWHT_Horizontal_AVX512(x.data, x.rows, x.cols, x.channels)".
// And "Process 16 channels simultaneously for the *same* spatial positions." (Packetized)
// This matches my analysis.
//
// Arguments:
// data: pointer to [H, W, C] buffer (batch 1)
// rows: H (64)
// cols: W (64)
// channels: C (64 or 192?) ZenithBlock is "192 -> 64 (Conv)".
// ZenithBlock operates on 64 channels.
//
// Implementation handles Packetized FWHT along W dimension (cols).
// Stride between W elements is `channels`.
inline void FWHT_Horizontal_AVX(float* data, int rows, int cols, int channels) {
    if (cols != 64) return; // Hardcoded for 64

    // Parallel over Rows and Channel-Packets
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < channels; c += 16) {
            float* row_ptr = data + (r * cols * channels) + c;

            // Check if we have 16 channels (AVX-512) or split for AVX2
            int rem = channels - c;

            // Packetized FWHT along W (size 64)
            // Loop h = 1, 2, 4, 8, 16, 32.
            // i loops 0..63.

            // We need to support AVX2 (8 floats) if AVX512 not present.
            // Or use AVX512 if present.

            #if defined(__AVX512F__)
            if (rem >= 16) {
                // Use AVX512 (16 channels)
                int h = 1;
                while (h < 64) {
                    for (int i = 0; i < 64; i += h * 2) {
                        for (int j = i; j < i + h; j++) {
                            float* x = row_ptr + (j * channels);
                            float* y = row_ptr + ((j + h) * channels);
                            Vec512 vx = Load512(x);
                            Vec512 vy = Load512(y);
                            Butterfly512(vx, vy);
                            Store512(x, vx);
                            Store512(y, vy);
                        }
                    }
                    h *= 2;
                }
                continue; // Done with this packet
            }
            #endif

            // Fallback / AVX2
            // Process in chunks of 8
            for (int sub_c = 0; sub_c < std::min(rem, 16); sub_c += 8) {
                int rem2 = std::min(rem, 16) - sub_c;
                if (rem2 >= 8) {
                    #ifdef __AVX2__
                    int h = 1;
                    while (h < 64) {
                        for (int i = 0; i < 64; i += h * 2) {
                            for (int j = i; j < i + h; j++) {
                                float* x = row_ptr + sub_c + (j * channels);
                                float* y = row_ptr + sub_c + ((j + h) * channels);
                                Vec256 vx = Load256(x);
                                Vec256 vy = Load256(y);
                                Butterfly256(vx, vy);
                                Store256(x, vx);
                                Store256(y, vy);
                            }
                        }
                        h *= 2;
                    }
                    #else
                    // Scalar fallback
                    int h = 1;
                    while (h < 64) {
                        for (int i = 0; i < 64; i += h * 2) {
                            for (int j = i; j < i + h; j++) {
                                float* x_ptr = row_ptr + sub_c + (j * channels);
                                float* y_ptr = row_ptr + sub_c + ((j + h) * channels);
                                for(int k=0; k<8; ++k) {
                                    float x = x_ptr[k];
                                    float y = y_ptr[k];
                                    x_ptr[k] = x + y;
                                    y_ptr[k] = x - y;
                                }
                            }
                        }
                        h *= 2;
                    }
                    #endif
                } else {
                    // Scalar tail
                    int h = 1;
                    while (h < 64) {
                        for (int i = 0; i < 64; i += h * 2) {
                            for (int j = i; j < i + h; j++) {
                                float* x_ptr = row_ptr + sub_c + (j * channels);
                                float* y_ptr = row_ptr + sub_c + ((j + h) * channels);
                                for(int k=0; k<rem2; ++k) {
                                    float x = x_ptr[k];
                                    float y = y_ptr[k];
                                    x_ptr[k] = x + y;
                                    y_ptr[k] = x - y;
                                }
                            }
                        }
                        h *= 2;
                    }
                }
            }
        }
    }

    // Normalization (1/sqrt(N) or 1/N?)
    // Usually FWHT is unitary if scaled by 1/sqrt(N).
    // ZenithBlock logic: "FWHT ... Mul ... FWHT".
    // If not unitary, we grow by N.
    // Plan says "multiply the final result by 1.0 / 64.0".
    // This implies we do Inverse via FWHT + Scale.
    // So this function is raw FWHT.
}

// Element-wise Multiply with Broadcast
// x: [H, W, C]. gate: [1, 1, C, Dim]? No.
// Plan: "gate_h; // Shape [1, 1, 64, Dim]"
// Wait. "Broadcasting vector".
// ZenithLiteBlock has 64 channels?
// If Channels=64.
// Gate should be [W] (for horizontal).
// If `gate_h` is "Shape [1, 1, 64, Dim]", maybe 64 is resolution?
// Gate H (horizontal) acts on W dimension.
// `x` is [H, W, C].
// We want to scale each column `w` by a factor `g[w]`.
// Wait, `Spectral Gating` usually means element-wise mul in spectral domain.
// If we transform W dimension to spectral, we have `[H, W_spec, C]`.
// We multiply by `gate[w]`.
// Gate should be shape `[W]`.
// Or `[W, C]`?
// Plan says: "gate_h ... Broadcasting vector".
// "Shape [1, 1, 64, Dim]"?
// If resolution is 64x64.
// Maybe [1, 1, 64, 1] (Gate per W index, broadcast over C)?
// Or [1, 1, 1, C] (Gate per C, broadcast over W)?
// "Separable FWHT + Gating".
// Row Mix: FWHT(Rows). Mul(Gate).
// If we Mix Rows (W dim), we are in W-spectral domain.
// Gate usually depends on Frequency (W index).
// So `gate` should be size `W`.
// Let's assume `gate` is `[W]`. Broadcast across `H` and `C`.
inline void ElementWiseMul_Broadcast(float* data, const float* gate, const std::vector<size_t>& shape) {
    // Shape [N, H, W, C]
    int H = shape[1];
    int W = shape[2];
    int C = shape[3];

    // Gate is [W].
    // data[h, w, c] *= gate[w].

    #pragma omp parallel for collapse(2)
    for(int h=0; h<H; ++h) {
        for(int w=0; w<W; ++w) {
            float g = gate[w];

            float* pixel = data + (h*W + w)*C;

            #if defined(__AVX512F__)
            Vec512 vg = _mm512_set1_ps(g);
            int c=0;
            for(; c+16<=C; c+=16) {
                Store512(pixel+c, Mul512(Load512(pixel+c), vg));
            }
            #elif defined(__AVX2__)
            Vec256 vg = _mm256_set1_ps(g);
            int c=0;
            for(; c+8<=C; c+=8) {
                Store256(pixel+c, Mul256(Load256(pixel+c), vg));
            }
            #else
            int c=0;
            #endif

            for(; c<C; ++c) {
                pixel[c] *= g;
            }
        }
    }
}

// Transpose Block 64x64
// Input: [H, W, C]. We want to transpose H and W?
// "Vertical Spectral Mix ... Block-Transpose ... FWHT Horizontal ... Transpose Back".
// Yes, Transpose (H, W).
// [H, W, C] -> [W, H, C].
// Cache blocking is crucial.
inline void Transpose_Block_64x64(const float* src, float* dst, int H, int W, int C) {
    const int TILE = 32;
    #pragma omp parallel for collapse(2)
    for(int h0=0; h0<H; h0+=TILE) {
        for(int w0=0; w0<W; w0+=TILE) {
            int h_max = std::min(h0+TILE, H);
            int w_max = std::min(w0+TILE, W);

            for(int h=h0; h<h_max; ++h) {
                for(int w=w0; w<w_max; ++w) {
                    const float* s = src + (h*W + w)*C;
                    float* d = dst + (w*H + h)*C;
                    std::memcpy(d, s, C * sizeof(float));
                }
            }
        }
    }
}

// SpaceToDepth Shuffle (AVX Optimized)
// NHWC [1, 512, 512, 3] -> [1, 64, 64, 192] (Block 8)
// Map (n, h, w, c) -> (n, h/8, w/8, (h%8)*8*C + (w%8)*C + c)
// Actually standard SpaceToDepth is:
// Output channel order: [block_size, block_size, C]
// channel = (ry * block_size + rx) * C + c
inline void SpaceToDepth_Shuffle(const float* src, float* dst, int H, int W, int C, int block_size) {
    int H_out = H / block_size;
    int W_out = W / block_size;
    int C_out = C * block_size * block_size;

    // Parallelize over output pixels
    #pragma omp parallel for collapse(2)
    for(int h=0; h<H_out; ++h) {
        for(int w=0; w<W_out; ++w) {
            float* out_pixel = dst + (h*W_out + w)*C_out;

            for(int ry=0; ry<block_size; ++ry) {
                for(int rx=0; rx<block_size; ++rx) {
                    int in_h = h * block_size + ry;
                    int in_w = w * block_size + rx;
                    const float* in_pixel = src + (in_h*W + in_w)*C;

                    // Copy C channels
                    // Destination offset: (ry * block_size + rx) * C
                    int out_c_offset = (ry * block_size + rx) * C;

                    std::memcpy(out_pixel + out_c_offset, in_pixel, C * sizeof(float));
                }
            }
        }
    }
}

// DepthToSpace (Inverse)
// Input: [H, W, C*bs*bs] -> [H*bs, W*bs, C]
inline void DepthToSpace_Shuffle(const float* src, float* dst, int H, int W, int C_out, int block_size) {
    int C_in = C_out * block_size * block_size;

    #pragma omp parallel for collapse(2)
    for(int h=0; h<H; ++h) {
        for(int w=0; w<W; ++w) {
            const float* in_pixel = src + (h*W + w)*C_in;

            for(int ry=0; ry<block_size; ++ry) {
                for(int rx=0; rx<block_size; ++rx) {
                    int out_h = h * block_size + ry;
                    int out_w = w * block_size + rx;
                    float* out_pixel = dst + (out_h*(W*block_size) + out_w)*C_out;

                    int in_c_offset = (ry * block_size + rx) * C_out;
                    std::memcpy(out_pixel, in_pixel + in_c_offset, C_out * sizeof(float));
                }
            }
        }
    }
}

} // namespace kernels
} // namespace dreidel
