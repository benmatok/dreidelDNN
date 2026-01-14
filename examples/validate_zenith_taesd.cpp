#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include "dreidel/models/ZenithTAESD.hpp"
#include "dreidel/hal/x86.hpp"
#include "dreidel/core/Tensor.hpp"
#include "dreidel/layers/Conv2D.hpp"
#include "dreidel/layers/OptimizedConv2D.hpp"
#include "dreidel/layers/Upscale2D.hpp"
#include "dreidel/layers/ZenithLiteBlock.hpp"

using namespace dreidel;

// Standard TAESD (Baseline)
// Architecture: Same as ZenithTAESD but ZenithLiteBlock is replaced by 2x Conv3x3 (Residual)
// Or just 1x Conv3x3 to be simpler/faster baseline?
// Standard ResNet block: Conv3x3 -> ReLU -> Conv3x3 -> Add.
// Let's implement a simple ResNetBlock for baseline.
template <typename T>
class ResNetBlock : public layers::Layer<T> {
public:
    ResNetBlock(size_t channels) {
        conv1_ = std::make_unique<layers::OptimizedConv2D<T>>(channels, channels, 3, 1, 1);
        conv2_ = std::make_unique<layers::OptimizedConv2D<T>>(channels, channels, 3, 1, 1);
    }
    Tensor<T> forward(const Tensor<T>& input) override {
        // Simplified ResNet (no ReLU for benchmark simplicity, just Ops)
        // Ideally should have ReLU.
        Tensor<T> x = conv1_->forward(input);
        x = conv2_->forward(x);
        // Add
        // In-place add
        T* out_ptr = x.data();
        const T* in_ptr = input.data();
        size_t size = x.size();
        for(size_t i=0; i<size; ++i) out_ptr[i] += in_ptr[i];
        return x;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override { return grad_output; }
    std::vector<Tensor<T>*> parameters() override { return {}; } // Dummy
    std::string name() const override { return "ResNetBlock"; }
private:
    std::unique_ptr<layers::OptimizedConv2D<T>> conv1_;
    std::unique_ptr<layers::OptimizedConv2D<T>> conv2_;
};

template <typename T>
class StandardTAESD : public layers::Layer<T> {
public:
    StandardTAESD(size_t in_channels = 3, size_t latent_channels = 4, size_t base_channels = 64, size_t height = 512, size_t width = 512)
        : H_(height), W_(width)
    {
        enc_stem_ = std::make_unique<layers::OptimizedConv2D<T>>(in_channels, base_channels, 3, 1, 1);
        enc_block1_ = std::make_unique<ResNetBlock<T>>(base_channels);
        enc_down1_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels, base_channels*2, 3, 2, 1);
        enc_block2_ = std::make_unique<ResNetBlock<T>>(base_channels*2);
        enc_down2_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*2, base_channels*4, 3, 2, 1);
        enc_block3_ = std::make_unique<ResNetBlock<T>>(base_channels*4);
        enc_out_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*4, latent_channels, 3, 2, 1);

        dec_in_ = std::make_unique<layers::OptimizedConv2D<T>>(latent_channels, base_channels*4, 1, 1, 0);
        dec_block1_ = std::make_unique<ResNetBlock<T>>(base_channels*4);
        dec_up1_ = std::make_unique<layers::Upscale2D<T>>(2);
        dec_conv_up1_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*4, base_channels*2, 3, 1, 1);
        dec_block2_ = std::make_unique<ResNetBlock<T>>(base_channels*2);
        dec_up2_ = std::make_unique<layers::Upscale2D<T>>(2);
        dec_conv_up2_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*2, base_channels, 3, 1, 1);
        dec_block3_ = std::make_unique<ResNetBlock<T>>(base_channels);
        dec_up3_ = std::make_unique<layers::Upscale2D<T>>(2);
        dec_out_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels, in_channels, 3, 1, 1);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = enc_stem_->forward(input);
        x = enc_block1_->forward(x);
        x = enc_down1_->forward(x);
        x = enc_block2_->forward(x);
        x = enc_down2_->forward(x);
        x = enc_block3_->forward(x);
        x = enc_out_->forward(x);

        x = dec_in_->forward(x);
        x = dec_block1_->forward(x);
        x = dec_up1_->forward(x);
        x = dec_conv_up1_->forward(x);
        x = dec_block2_->forward(x);
        x = dec_up2_->forward(x);
        x = dec_conv_up2_->forward(x);
        x = dec_block3_->forward(x);
        x = dec_up3_->forward(x);
        x = dec_out_->forward(x);
        return x;
    }

    // Stubs
    Tensor<T> backward(const Tensor<T>& grad_output) override { return grad_output; }
    std::string name() const override { return "StandardTAESD"; }

private:
    size_t H_, W_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_stem_;
    std::unique_ptr<ResNetBlock<T>> enc_block1_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_down1_;
    std::unique_ptr<ResNetBlock<T>> enc_block2_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_down2_;
    std::unique_ptr<ResNetBlock<T>> enc_block3_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_out_;

    std::unique_ptr<layers::OptimizedConv2D<T>> dec_in_;
    std::unique_ptr<ResNetBlock<T>> dec_block1_;
    std::unique_ptr<layers::Upscale2D<T>> dec_up1_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_conv_up1_;
    std::unique_ptr<ResNetBlock<T>> dec_block2_;
    std::unique_ptr<layers::Upscale2D<T>> dec_up2_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_conv_up2_;
    std::unique_ptr<ResNetBlock<T>> dec_block3_;
    std::unique_ptr<layers::Upscale2D<T>> dec_up3_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_out_;
};


// Helper to check FWHT Invertibility
void check_fwht_invertibility() {
    std::cout << "Checking FWHT Invertibility..." << std::endl;
    size_t N = 128; // Spatial
    size_t C = 64;  // Channels

    std::vector<float> data(N * C);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(auto& v : data) v = dist(rng);

    std::vector<float> original = data;

    // FWHT
    hal::x86::fwht_1d_vectorized_avx2(data.data(), N, C);

    // IFWHT (Same kernel)
    hal::x86::fwht_1d_vectorized_avx2(data.data(), N, C);

    // Scale by 1/N
    float scale = 1.0f / N;
    float max_err = 0.0f;
    for(size_t i=0; i<data.size(); ++i) {
        float val = data[i] * scale;
        float diff = std::abs(val - original[i]);
        if(diff > max_err) max_err = diff;
    }

    std::cout << "FWHT Max Error: " << max_err << std::endl;
    if (max_err < 1e-4) std::cout << "FWHT Invertibility: PASS" << std::endl;
    else std::cout << "FWHT Invertibility: FAIL" << std::endl;
}

int main() {
    std::cout << "=== Zenith-TAESD Validation & Benchmark ===\n" << std::endl;

    // 1. Check Kernels
    check_fwht_invertibility();

    // 2. Instantiate Model
    size_t H = 512;
    size_t W = 512;
    size_t C = 3;
    models::ZenithTAESD<float> model_zenith(C, 4, 64, H, W);
    StandardTAESD<float> model_baseline(C, 4, 64, H, W);

    std::cout << "Models instantiated." << std::endl;

    Tensor<float> input({1, H, W, C});
    input.random(0.0f, 1.0f);

    int iterations = 3; // Reduced for quick feedback

    // --- Zenith Benchmark ---
    std::cout << "\nBenchmarking Zenith-TAESD (512x512, " << iterations << " iters)..." << std::endl;
    layers::ZenithLiteBlock<float>::reset_timers();
    layers::OptimizedConv2D<float>::reset_timers();

    auto start_z = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        model_zenith.forward(input);
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double elapsed_z = std::chrono::duration<double, std::milli>(end_z - start_z).count() / iterations;

    std::cout << "Zenith Avg Latency: " << elapsed_z << " ms" << std::endl;
    layers::ZenithLiteBlock<float>::print_timers();
    std::cout << "Zenith Conv2D Total: ";
    layers::OptimizedConv2D<float>::print_timers();

    // --- Baseline Benchmark ---
    std::cout << "\nBenchmarking Standard-TAESD (512x512, " << iterations << " iters)..." << std::endl;
    layers::OptimizedConv2D<float>::reset_timers();

    auto start_b = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        model_baseline.forward(input);
    }
    auto end_b = std::chrono::high_resolution_clock::now();
    double elapsed_b = std::chrono::duration<double, std::milli>(end_b - start_b).count() / iterations;

    std::cout << "Baseline Avg Latency: " << elapsed_b << " ms" << std::endl;
    std::cout << "Baseline Conv2D Total: ";
    layers::OptimizedConv2D<float>::print_timers();

    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Speedup (Zenith vs Baseline): " << elapsed_b / elapsed_z << "x" << std::endl;

    // FLOPs Estimation
    // Zenith: ZenithLite blocks (minimal) + Conv2D layers.
    // Major Conv2Ds:
    // Stem: 3->64, 3x3, 512x512.
    // Down1: 64->128, 3x3, 256x256.
    // Down2: 128->256, 3x3, 128x128.
    // Out: 256->4, 3x3, 64x64.
    // Dec In: 4->256, 1x1, 64x64.
    // Dec Up1: 256->128, 3x3, 128x128.
    // Dec Up2: 128->64, 3x3, 256x256.
    // Dec Out: 64->3, 3x3, 512x512.

    double flops = 0;
    auto add_conv = [&](double h, double w, double cin, double cout, double k) {
        flops += h * w * cin * cout * k * k * 2.0; // MAC = 2 ops
    };

    // Encoder
    add_conv(512, 512, 3, 64, 3);
    add_conv(256, 256, 64, 128, 3);
    add_conv(128, 128, 128, 256, 3);
    add_conv(64, 64, 256, 4, 3);

    // Decoder
    add_conv(64, 64, 4, 256, 1);
    add_conv(128, 128, 256, 128, 3);
    add_conv(256, 256, 128, 64, 3);
    add_conv(512, 512, 64, 3, 3); // 3->3 or 64->3? Base->In. 64->3.

    // Plus ZenithLite (Compress/Expand)
    // 6 blocks.
    // Approx Compress (1x1): H*W*C*C/4 * 2. Expand: H*W*C/4*C * 2.
    // Negligible compared to Conv2D?
    // Let's add them.
    auto add_zenith = [&](double h, double w, double c) {
        double inner = c/4;
        flops += h * w * c * inner * 1 * 1 * 2 * 2; // Compress + Expand
    };
    add_zenith(512, 512, 64);
    add_zenith(256, 256, 128);
    add_zenith(128, 128, 256);
    add_zenith(64, 64, 256); // Block1 dec
    add_zenith(128, 128, 128); // Block2 dec
    add_zenith(256, 256, 64); // Block3 dec

    std::cout << "\n=== Theoretical Analysis ===" << std::endl;
    std::cout << "Estimated FLOPs per Image: " << flops / 1e9 << " GFLOPs" << std::endl;
    std::cout << "Zenith Actual Performance: " << (flops / 1e9) / (elapsed_z / 1000.0) << " GFLOPs/s" << std::endl;
    std::cout << "Baseline Actual Performance: " << (flops / 1e9) / (elapsed_b / 1000.0) << " GFLOPs/s" << std::endl; // Baseline has similar FLOPs (ResNet vs Zenith) roughly
    std::cout << "Target Performance (CPU AVX2): >50 GFLOPs/s" << std::endl;

    return 0;
}
