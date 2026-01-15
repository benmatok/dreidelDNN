#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cstdlib>

// ZenithNano_Pipelined.cpp
// Benchmarking & Profiling for High-Performance Inference
// Target: < 3ms per image (64x64x64 Block Latency)

// Configuration
const int H = 64;
const int W = 64;
const int C = 64;
const int ELEMENTS = H * W * C;
const int BLOCK_SIZE_C = 64; // Process all channels per pixel

// Utils
void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
}

// Data Buffers (Aligned for AVX)
struct ImageBuffers {
    float* float_data;
    short* int_data; // Int16 representation

    ImageBuffers() {
        float_data = (float*)aligned_alloc_wrapper(32, ELEMENTS * sizeof(float));
        int_data = (short*)aligned_alloc_wrapper(32, ELEMENTS * sizeof(short));

        // Init
        for(int i=0; i<ELEMENTS; ++i) {
            float_data[i] = 1.0f;
            int_data[i] = 1;
        }
    }

    ~ImageBuffers() {
        free(float_data);
        free(int_data);
    }
};

// --- Serial Kernel (Float Only) ---
// Processes one pixel (64 channels)
// Returns sum to prevent DCE
float process_pixel_serial(const float* in, float* out, __m256 w, __m256 b) {
    // Load 64 channels -> 8 AVX registers
    __m256 r0 = _mm256_load_ps(in + 0);
    __m256 r1 = _mm256_load_ps(in + 8);
    __m256 r2 = _mm256_load_ps(in + 16);
    __m256 r3 = _mm256_load_ps(in + 24);
    __m256 r4 = _mm256_load_ps(in + 32);
    __m256 r5 = _mm256_load_ps(in + 40);
    __m256 r6 = _mm256_load_ps(in + 48);
    __m256 r7 = _mm256_load_ps(in + 56);

    // Simulated FWHT (Add/Sub) - Float
    // Just 1 pass simulation (butterfly)
    // r0+r1, r0-r1, etc.
    __m256 t0 = _mm256_add_ps(r0, r1);
    __m256 t1 = _mm256_sub_ps(r0, r1);
    r0 = t0; r1 = t1;

    // ... Assume more passes ...
    for(int k=0; k<4; ++k) { // 4 passes
        r0 = _mm256_add_ps(r0, r1);
        r1 = _mm256_sub_ps(r0, r1);
    }

    // Simulated Conv (FMA)
    // 64 -> 64 channel mix simulation (Dense Dot Prod would be loop over input channels)
    // Here we simulate the *output* accumulator update.
    // For 1x1 Conv, we accumulate (in * weight) into (out).
    // Let's assume we do 64 MACs per output channel.
    // That's 64*64 ops.
    // This kernel processes ONE pixel.

    // To match the load of 16M ops per image:
    // Image has 4096 pixels.
    // 16M / 4096 = 4000 ops per pixel.
    // We have 8 registers. 4000 ops / 8 = 500 loops.
    // That's heavy.

    // Let's unroll decently.
    for(int i=0; i<64; ++i) { // 64 input channels
        r0 = _mm256_fmadd_ps(r0, w, b);
        r1 = _mm256_fmadd_ps(r1, w, b);
        r2 = _mm256_fmadd_ps(r2, w, b);
        r3 = _mm256_fmadd_ps(r3, w, b);
        r4 = _mm256_fmadd_ps(r4, w, b);
        r5 = _mm256_fmadd_ps(r5, w, b);
        r6 = _mm256_fmadd_ps(r6, w, b);
        r7 = _mm256_fmadd_ps(r7, w, b);
    }

    // Store
    _mm256_store_ps(out + 0, r0);
    _mm256_store_ps(out + 8, r1);
    _mm256_store_ps(out + 16, r2);
    _mm256_store_ps(out + 24, r3);
    _mm256_store_ps(out + 32, r4);
    _mm256_store_ps(out + 40, r5);
    _mm256_store_ps(out + 48, r6);
    _mm256_store_ps(out + 56, r7);

    return ((float*)&r0)[0];
}

void run_serial(ImageBuffers& buf, int iterations) {
    __m256 w = _mm256_set1_ps(0.01f);
    __m256 b = _mm256_set1_ps(0.001f);

    auto start = std::chrono::high_resolution_clock::now();

    for(int iter=0; iter<iterations; ++iter) {
        // Iterate over pixels
        #pragma omp parallel for collapse(2)
        for(int h=0; h<H; ++h) {
            for(int w_idx=0; w_idx<W; ++w_idx) {
                int offset = (h * W + w_idx) * C;
                process_pixel_serial(buf.float_data + offset, buf.float_data + offset, w, b);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Serial Float: " << time_ms / iterations << " ms/image" << std::endl;
}

// --- Pipelined Kernel (Hybrid) ---
// Processes TWO pixels at once:
// Pixel A: Float Channel Mix (Conv)
// Pixel B: Int Spatial Mix (FWHT)
// We need to fetch Pixel B from int_data.
// And Pixel A from float_data.
void process_pixels_pipelined(const float* f_in, float* f_out,
                              const short* i_in, short* i_out,
                              __m256 w_reg, __m256 b_reg)
{
    // Load Float Pixel (8 regs)
    __m256 f0 = _mm256_load_ps(f_in + 0);
    __m256 f1 = _mm256_load_ps(f_in + 8);
    __m256 f2 = _mm256_load_ps(f_in + 16);
    __m256 f3 = _mm256_load_ps(f_in + 24);
    __m256 f4 = _mm256_load_ps(f_in + 32);
    __m256 f5 = _mm256_load_ps(f_in + 40);
    __m256 f6 = _mm256_load_ps(f_in + 48);
    __m256 f7 = _mm256_load_ps(f_in + 56);

    // Load Int Pixel (4 regs for 64 shorts)
    // 64 shorts = 4 * 128 bits? No.
    // 1 AVX2 reg = 256 bits = 16 shorts.
    // 64 shorts = 4 registers. Correct.
    __m256i i0 = _mm256_load_si256((const __m256i*)(i_in + 0));
    __m256i i1 = _mm256_load_si256((const __m256i*)(i_in + 16));
    __m256i i2 = _mm256_load_si256((const __m256i*)(i_in + 32));
    __m256i i3 = _mm256_load_si256((const __m256i*)(i_in + 48));

    // Interleaved Loop
    // We have 64 loops of FMAs for Float.
    // We have ~4-8 loops of Adds for Int.
    // We distribute Int work into the Float loop.

    // Int workload: 4 passes * 4 regs = 16 ops.
    // Float workload: 64 passes * 8 regs = 512 ops.
    // Ratio: 1 : 32.
    // Int is very light compared to Conv.
    // So Int will be completely hidden.

    // We do Int ops in the first few iterations.

    // Pass 1 Int (Butterfly stride 1)
    // i0 += i1; i0 -= i1; ... logic needs care for shuffle.
    // Simple Add/Sub simulation:
    __m256i t_i;

    for(int k=0; k<64; ++k) {
        // --- Float Work ---
        f0 = _mm256_fmadd_ps(f0, w_reg, b_reg);
        f1 = _mm256_fmadd_ps(f1, w_reg, b_reg);

        // --- Int Work (Injected in first 16 iters) ---
        if (k < 4) { // 4 Passes
            i0 = _mm256_add_epi16(i0, i1);
            i2 = _mm256_add_epi16(i2, i3);
        } else if (k < 8) {
             i1 = _mm256_sub_epi16(i0, i1);
             i3 = _mm256_sub_epi16(i2, i3);
        }

        // Continue Float
        f2 = _mm256_fmadd_ps(f2, w_reg, b_reg);
        f3 = _mm256_fmadd_ps(f3, w_reg, b_reg);
        f4 = _mm256_fmadd_ps(f4, w_reg, b_reg);
        f5 = _mm256_fmadd_ps(f5, w_reg, b_reg);
        f6 = _mm256_fmadd_ps(f6, w_reg, b_reg);
        f7 = _mm256_fmadd_ps(f7, w_reg, b_reg);
    }

    // Store
    _mm256_store_ps(f_out + 0, f0);
    _mm256_store_ps(f_out + 8, f1);
    _mm256_store_ps(f_out + 16, f2);
    _mm256_store_ps(f_out + 24, f3);
    _mm256_store_ps(f_out + 32, f4);
    _mm256_store_ps(f_out + 40, f5);
    _mm256_store_ps(f_out + 48, f6);
    _mm256_store_ps(f_out + 56, f7);

    _mm256_store_si256((__m256i*)(i_out + 0), i0);
    _mm256_store_si256((__m256i*)(i_out + 16), i1);
    _mm256_store_si256((__m256i*)(i_out + 32), i2);
    _mm256_store_si256((__m256i*)(i_out + 48), i3);
}

void run_pipelined(ImageBuffers& buf, int iterations) {
    __m256 w = _mm256_set1_ps(0.01f);
    __m256 b = _mm256_set1_ps(0.001f);

    auto start = std::chrono::high_resolution_clock::now();

    for(int iter=0; iter<iterations; ++iter) {
        // Iterate over pixels
        // To use pipelining effectively, we pair the current pixel (Float Conv)
        // with the *next* pixel (Int FWHT) or same pixel?
        // Let's assume we do 1:1 mapping.

        #pragma omp parallel for collapse(2)
        for(int h=0; h<H; ++h) {
            for(int w_idx=0; w_idx<W; ++w_idx) {
                int offset = (h * W + w_idx) * C;
                // Float acts on buf.float_data
                // Int acts on buf.int_data (simulated spatial view)
                process_pixels_pipelined(
                    buf.float_data + offset, buf.float_data + offset,
                    buf.int_data + offset, buf.int_data + offset,
                    w, b
                );
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Pipelined Hybrid: " << time_ms / iterations << " ms/image" << std::endl;
}

int main() {
    std::cout << "ZenithNano Performance Benchmark" << std::endl;
    std::cout << "Image Size: " << H << "x" << W << "x" << C << std::endl;

    ImageBuffers buf;

    // Warmup
    run_serial(buf, 10);

    int benchmark_iters = 100;

    run_serial(buf, benchmark_iters);
    run_pipelined(buf, benchmark_iters);

    return 0;
}
