#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>

// Jules Architecture
#include "dreidel/jules/Agent.hpp"
#include "dreidel/core/Arena.hpp"
#include "dreidel/utils/WaveletGen.hpp"
#include "dreidel/utils/MatrixOps.hpp"

// Define static memory for the Agent (2MB)
static uint8_t NETWORK_WORKSPACE[2 * 1024 * 1024];

// Dimensions
constexpr size_t INPUT_DIM = 1024;
constexpr size_t LATENT_DIM = 64;

namespace dreidel {
namespace jules {

template <size_t IN_DIM, size_t LATENT_DIM>
class JulesAutoencoder : public Agent {
public:
    SpectralBlock<IN_DIM> enc_spectral;
    SparseBlock<IN_DIM, LATENT_DIM, 4> enc_sparse;
    DenseLUT<LATENT_DIM, IN_DIM> dec_dense;

    core::Arena* workspace_;

    JulesAutoencoder(core::Arena* workspace) : workspace_(workspace) {}

    void init() override {
        quant::APoT::init();
    }

    // Forward Pass (Inference) - Zero Alloc
    void step(const float* input_ptr, float* output_ptr) override {
        workspace_->reset();

        // Input View
        float* buf_in = workspace_->allocate<float>(IN_DIM);
        std::copy(input_ptr, input_ptr + IN_DIM, buf_in);
        core::TensorView<float> view_in(buf_in, IN_DIM);

        // 1. Encoder: Spectral
        float* buf_spectral = workspace_->allocate<float>(IN_DIM);
        core::TensorView<float> view_spectral(buf_spectral, IN_DIM);
        enc_spectral.forward(view_in, view_spectral);

        // 2. Encoder: Sparse
        float* buf_latent = workspace_->allocate<float>(LATENT_DIM);
        core::TensorView<float> view_latent(buf_latent, LATENT_DIM);
        enc_sparse.forward(view_spectral, view_latent);

        // 3. Decoder: DenseLUT
        core::TensorView<float> view_out(output_ptr, IN_DIM);
        dec_dense.forward(view_latent, view_out);
    }

    // Calibration Helper (Offline Training) - Uses Arena
    void calibrate(const std::vector<float>& batch_X, size_t batch_size) {
        std::cout << "Calibrating Decoder (Zero-Alloc)..." << std::endl;
        workspace_->reset();

        // Allocate H and Matrix Ops workspace in Arena
        // H: (B, L)
        float* batch_H = workspace_->allocate<float>(batch_size * LATENT_DIM);

        for(size_t i=0; i<batch_size; ++i) {
            // Re-use small buffers?
            // We can't reset arena inside loop as batch_H is there.
            // So we need small scratch for forward pass.
            // Since Forward allocs ~1024 floats, we can afford it inside loop if Arena is big enough.
            // 2MB is plenty for 500 samples * 64 floats (128KB) + scratch.

            // Wait, step() resets workspace! We can't use step().
            // We must call layers manually and manage scratch carefully.
            // Let's alloc scratch buffers ONCE at top of loop logic (conceptually).
            // Actually, we can just alloc buffers after batch_H.

            float* input_buf = workspace_->allocate<float>(IN_DIM);
            float* spectral_buf = workspace_->allocate<float>(IN_DIM);
            float* latent_buf = workspace_->allocate<float>(LATENT_DIM);

            // Copy input
            std::copy(batch_X.begin() + i*IN_DIM, batch_X.begin() + (i+1)*IN_DIM, input_buf);

            core::TensorView<float> v_in(input_buf, IN_DIM);
            core::TensorView<float> v_spec(spectral_buf, IN_DIM);
            core::TensorView<float> v_lat(latent_buf, LATENT_DIM);

            enc_spectral.forward(v_in, v_spec);
            enc_sparse.forward(v_spec, v_lat);

            // Copy to Batch H
            std::copy(latent_buf, latent_buf + LATENT_DIM, batch_H + i*LATENT_DIM);

            // Note: We leak the scratch buffers here inside the arena.
            // 500 * (1024+1024+64) * 4 bytes ~ 4MB.
            // Arena is 2MB. Oops.
            // We need to Reuse scratch buffers.
            // But allocate gives linear pointer.
            // Simple fix: Alloc scratch buffers ONCE before loop.
            // But we can't 'free' them to use for next iter?
            // We just overwrite them.
            // So:
            // 1. Alloc Batch H.
            // 2. Alloc Scratch (Input, Spec, Latent).
            // 3. Loop: Use Scratch -> Copy Latent to Batch H.
            // Correct.
        }

        // To implement "Reuse", we just use pointers.
        // BUT wait, previous logic inside loop allocated NEW buffers.
        // Let's refactor.
    }

    // Improved Calibrate with Scratch Reuse
    void calibrate_optimized(const std::vector<float>& batch_X, size_t batch_size) {
        std::cout << "Calibrating Decoder (Optimized Arena Usage)..." << std::endl;
        workspace_->reset();

        // 1. Alloc Permanent Storage for H
        float* batch_H = workspace_->allocate<float>(batch_size * LATENT_DIM);

        // 2. Alloc Scratch Buffers (Reused)
        float* input_buf = workspace_->allocate<float>(IN_DIM);
        float* spectral_buf = workspace_->allocate<float>(IN_DIM);
        float* latent_buf = workspace_->allocate<float>(LATENT_DIM);

        // 3. Collect H
        for(size_t i=0; i<batch_size; ++i) {
             std::copy(batch_X.begin() + i*IN_DIM, batch_X.begin() + (i+1)*IN_DIM, input_buf);

             core::TensorView<float> v_in(input_buf, IN_DIM);
             core::TensorView<float> v_spec(spectral_buf, IN_DIM);
             core::TensorView<float> v_lat(latent_buf, LATENT_DIM);

             enc_spectral.forward(v_in, v_spec);
             enc_sparse.forward(v_spec, v_lat);

             std::copy(latent_buf, latent_buf + LATENT_DIM, batch_H + i*LATENT_DIM);
        }

        // 4. Solve Ridge Regression using Arena for Matrices
        // A = H^T H (L x L)
        float* A = workspace_->allocate<float>(LATENT_DIM * LATENT_DIM);
        std::fill(A, A + LATENT_DIM * LATENT_DIM, 0.0f);

        // Compute A
        for(size_t r=0; r<LATENT_DIM; ++r) {
            for(size_t c=0; c<LATENT_DIM; ++c) {
                float sum = 0.0f;
                for(size_t k=0; k<batch_size; ++k) {
                    sum += batch_H[k*LATENT_DIM + r] * batch_H[k*LATENT_DIM + c];
                }
                A[r*LATENT_DIM + c] = sum;
            }
        }

        // Regularization
        for(size_t i=0; i<LATENT_DIM; ++i) A[i*LATENT_DIM + i] += 1.0f;

        // Workspace for Inversion (Augmented matrix 2*L*L)
        float* inv_workspace = workspace_->allocate<float>(LATENT_DIM * 2 * LATENT_DIM);

        // Invert
        if (!utils::invert_matrix_ptr(A, (int)LATENT_DIM, inv_workspace)) {
            std::cerr << "Inversion failed." << std::endl;
            return;
        }

        // B = H^T X (L x D)
        // Alloc B
        float* B_mat = workspace_->allocate<float>(LATENT_DIM * IN_DIM);
        // Note: Check arena capacity.
        // H: 500*64*4 = 128KB
        // A: 64*64*4 = 16KB
        // Inv: 64*128*4 = 32KB
        // B: 64*1024*4 = 256KB
        // Total so far < 1MB. Safe.

        for(size_t r=0; r<LATENT_DIM; ++r) {
            for(size_t c=0; c<IN_DIM; ++c) {
                float sum = 0.0f;
                for(size_t k=0; k<batch_size; ++k) {
                    sum += batch_H[k*LATENT_DIM + r] * batch_X[k*IN_DIM + c];
                }
                B_mat[r*IN_DIM + c] = sum;
            }
        }

        // W = A_inv * B (L x D)
        // We can compute W row by row and quantize immediately to save memory if needed,
        // but we have space.
        float* W = workspace_->allocate<float>(LATENT_DIM * IN_DIM);

        for(size_t r=0; r<LATENT_DIM; ++r) {
            for(size_t c=0; c<IN_DIM; ++c) {
                float sum = 0.0f;
                for(size_t k=0; k<LATENT_DIM; ++k) {
                    sum += A[r*LATENT_DIM + k] * B_mat[k*IN_DIM + c];
                }
                W[r*IN_DIM + c] = sum;
            }
        }

        // Quantize
        for(size_t i=0; i<LATENT_DIM * IN_DIM; ++i) {
            dec_dense.weights[i] = quant::APoT::quantize(W[i]);
        }
        std::cout << "Calibration Complete." << std::endl;
    }
};

} // namespace jules
} // namespace dreidel

// Define Global Agent
static dreidel::core::Arena arena(NETWORK_WORKSPACE, sizeof(NETWORK_WORKSPACE));
static dreidel::jules::JulesAutoencoder<1024, 64> agent(&arena);

int main() {
    std::cout << "=== Project Jules: Autoencoder Demo ===" << std::endl;
    std::cout << "Architecture: Wavelet(1024) -> Spectral -> Sparse -> Latent(64) -> DenseLUT(APoT) -> Out(1024)" << std::endl;

    // 1. Generate Training Data
    size_t train_size = 500;
    std::cout << "Generating " << train_size << " wavelets for calibration..." << std::endl;

    dreidel::Tensor<float> train_data({train_size, 1024});
    dreidel::utils::generate_mixed_wavelets(train_data, train_size, 1024);

    std::vector<float> train_vec(train_data.data(), train_data.data() + train_data.size());

    // 2. Init & Calibrate
    agent.init();
    agent.calibrate_optimized(train_vec, train_size);

    // 3. Inference & Validation
    size_t val_size = 1;
    dreidel::Tensor<float> val_data({val_size, 1024});
    dreidel::utils::generate_mixed_wavelets(val_data, val_size, 1024);

    float output_buf[1024];

    auto start = std::chrono::high_resolution_clock::now();
    agent.step(val_data.data(), output_buf);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Inference Time: " << std::chrono::duration<double>(end - start).count() * 1000.0 << " ms" << std::endl;

    // 4. Export for Visualization
    std::ofstream csv("jules_autoencoder_results.csv");
    csv << "t,input,reconstructed\n";
    for(size_t t=0; t<1024; ++t) {
        csv << t << "," << val_data.data()[t] << "," << output_buf[t] << "\n";
    }
    csv.close();

    // 5. Generate SVG
    std::cout << "Generating SVG..." << std::endl;
    std::ofstream svg("jules_reconstruction.svg");
    if(svg.is_open()) {
        double width = 800, height = 400;
        double padding = 50;
        svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\">\n";
        svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\" />\n";

        float min_val = 1e9, max_val = -1e9;
        for(size_t t=0; t<1024; ++t) {
            float v1 = val_data.data()[t];
            float v2 = output_buf[t];
            if(v1 < min_val) min_val = v1; if(v1 > max_val) max_val = v1;
            if(v2 < min_val) min_val = v2; if(v2 > max_val) max_val = v2;
        }
        float range = max_val - min_val;
        if(range < 1e-5) range = 1.0;

        auto map_x = [&](size_t i) { return padding + (double)i/1024.0 * (width - 2*padding); };
        auto map_y = [&](float v) { return height - padding - (v - min_val)/range * (height - 2*padding); };

        svg << "<path d=\"M";
        for(size_t t=0; t<1024; ++t) {
            svg << map_x(t) << " " << map_y(val_data.data()[t]);
            if(t < 1023) svg << " L ";
        }
        svg << "\" fill=\"none\" stroke=\"black\" stroke-width=\"2\" opacity=\"0.5\" />\n";

        svg << "<path d=\"M";
        for(size_t t=0; t<1024; ++t) {
            svg << map_x(t) << " " << map_y(output_buf[t]);
            if(t < 1023) svg << " L ";
        }
        svg << "\" fill=\"none\" stroke=\"red\" stroke-width=\"2\" />\n";

        svg << "<text x=\"" << width-150 << "\" y=\"30\" fill=\"black\">Input (Black)</text>\n";
        svg << "<text x=\"" << width-150 << "\" y=\"50\" fill=\"red\">Recon (Red)</text>\n";

        svg << "</svg>\n";
    }

    std::cout << "Done. Check jules_reconstruction.svg" << std::endl;
    return 0;
}
