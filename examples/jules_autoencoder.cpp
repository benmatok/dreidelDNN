#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>

// Jules Architecture
#include "dreidel/jules/Agent.hpp"
#include "dreidel/jules/FusedJulesBlock.hpp" // New Block
#include "dreidel/core/Arena.hpp"
#include "dreidel/utils/WaveletGen.hpp"
#include "dreidel/utils/MatrixOps.hpp"

// Define static memory for the Agent (16MB to support 1024x1024 matrix ops)
static uint8_t NETWORK_WORKSPACE[16 * 1024 * 1024];

// Dimensions
constexpr size_t INPUT_DIM = 1024;
constexpr size_t HIDDEN_DIM = 1024;
constexpr size_t LATENT_DIM = 64;
constexpr size_t NUM_LAYERS = 2; // Depth of Encoder/Decoder

namespace dreidel {
namespace jules {

template <size_t IN_DIM, size_t LATENT_DIM>
class DeepJulesAutoencoder : public Agent {
public:
    // Encoder Stack
    FusedJulesBlock<IN_DIM, IN_DIM> enc_layer1;
    FusedJulesBlock<IN_DIM, IN_DIM> enc_layer2;
    DenseLUT<IN_DIM, LATENT_DIM> enc_bottleneck;

    // Decoder Stack
    DenseLUT<LATENT_DIM, IN_DIM> dec_proj;
    FusedJulesBlock<IN_DIM, IN_DIM> dec_layer1;
    FusedJulesBlock<IN_DIM, IN_DIM> dec_layer2;

    // Final Readout (Calibrated)
    DenseLUT<IN_DIM, IN_DIM> dec_readout;

    core::Arena* workspace_;

    DeepJulesAutoencoder(core::Arena* workspace) : workspace_(workspace) {}

    void init() override {
        quant::APoT::init();
    }

    // Forward Pass
    void step(const float* input_ptr, float* output_ptr) override {
        workspace_->reset();

        // 1. Copy Input
        float* buf_curr = workspace_->allocate<float>(IN_DIM);
        std::copy(input_ptr, input_ptr + IN_DIM, buf_curr);
        core::TensorView<float> curr(buf_curr, IN_DIM);

        // 2. Encoder Layers (Ping Pong)
        float* buf_next = workspace_->allocate<float>(IN_DIM);
        core::TensorView<float> next(buf_next, IN_DIM);

        // L1
        enc_layer1.forward(curr, next, workspace_);

        // L2 (Next -> Curr)
        core::TensorView<float> next_out(buf_curr, IN_DIM);
        enc_layer2.forward(next, next_out, workspace_);

        // 3. Bottleneck
        float* buf_lat = workspace_->allocate<float>(LATENT_DIM);
        core::TensorView<float> lat(buf_lat, LATENT_DIM);
        enc_bottleneck.forward(next_out, lat);

        // 4. Decoder Projection
        core::TensorView<float> dec_in(buf_next, IN_DIM);
        dec_proj.forward(lat, dec_in);

        // 5. Decoder Layers
        core::TensorView<float> dec_l1_out(buf_curr, IN_DIM);
        dec_layer1.forward(dec_in, dec_l1_out, workspace_);

        core::TensorView<float> dec_l2_out(buf_next, IN_DIM);
        dec_layer2.forward(dec_l1_out, dec_l2_out, workspace_);

        // 6. Readout
        core::TensorView<float> out_view(output_ptr, IN_DIM);
        dec_readout.forward(dec_l2_out, out_view);
    }

    // Calibrate Readout
    void calibrate(const std::vector<float>& batch_X, size_t batch_size) {
        std::cout << "Calibrating Deep Decoder..." << std::endl;

        // Temporarily collect features in Heap vector to simplify Arena reuse logic
        // for this demo.
        std::vector<float> vec_F(batch_size * IN_DIM);

        for(size_t i=0; i<batch_size; ++i) {
             workspace_->reset();

             // Run forward to Dec L2
             // Copy Input
            float* buf_curr = workspace_->allocate<float>(IN_DIM);
            std::copy(batch_X.begin() + i*IN_DIM, batch_X.begin() + (i+1)*IN_DIM, buf_curr);
            core::TensorView<float> curr(buf_curr, IN_DIM);

            float* buf_next = workspace_->allocate<float>(IN_DIM);
            core::TensorView<float> next(buf_next, IN_DIM);
            enc_layer1.forward(curr, next, workspace_);

            core::TensorView<float> next_out(buf_curr, IN_DIM);
            enc_layer2.forward(next, next_out, workspace_);

            float* buf_lat = workspace_->allocate<float>(LATENT_DIM);
            core::TensorView<float> lat(buf_lat, LATENT_DIM);
            enc_bottleneck.forward(next_out, lat);

            core::TensorView<float> dec_in(buf_next, IN_DIM);
            dec_proj.forward(lat, dec_in);

            core::TensorView<float> dec_l1_out(buf_curr, IN_DIM);
            dec_layer1.forward(dec_in, dec_l1_out, workspace_);

            core::TensorView<float> dec_l2_out(buf_next, IN_DIM);
            dec_layer2.forward(dec_l1_out, dec_l2_out, workspace_);

            // Copy dec_l2_out to vec_F
            std::copy(dec_l2_out.data(), dec_l2_out.data() + IN_DIM, vec_F.begin() + i*IN_DIM);
        }

        // Now Solve on Arena
        workspace_->reset();

        // F
        float* A_mat = workspace_->allocate<float>(batch_size * IN_DIM);
        std::copy(vec_F.begin(), vec_F.end(), A_mat);

        // Cov = F^T F (D x D) = 4MB
        float* Cov = workspace_->allocate<float>(IN_DIM * IN_DIM);
        std::fill(Cov, Cov + IN_DIM * IN_DIM, 0.0f);

        for(size_t r=0; r<IN_DIM; ++r) {
            for(size_t c=0; c<IN_DIM; ++c) {
                float sum = 0.0f;
                for(size_t k=0; k<batch_size; ++k) {
                    sum += A_mat[k*IN_DIM + r] * A_mat[k*IN_DIM + c];
                }
                Cov[r*IN_DIM + c] = sum;
            }
        }

        // Reg
        for(size_t i=0; i<IN_DIM; ++i) Cov[i*IN_DIM + i] += 10.0f;

        // Invert (needs 2*D*D workspace = 8MB)
        float* inv_wk = workspace_->allocate<float>(IN_DIM * 2 * IN_DIM);
        if (!utils::invert_matrix_ptr(Cov, (int)IN_DIM, inv_wk)) {
             std::cerr << "Inversion failed" << std::endl;
             return;
        }

        // Cross = F^T X (D x D) = 4MB
        // NOTE: We might OOM with 16MB.
        // 4 (Cov) + 8 (Inv) = 12.
        // We need 4 for Cross and 4 for W_T. Total ~20MB?
        // Let's compute Cross BEFORE Invert and store it?
        // No, we need Cov and Cross to compute W.
        // We can reuse A_mat buffer (batch_size*D = 500*1024*4 = 2MB).
        // Total: 2 (A) + 4 (Cov) + 8 (Inv) + 4 (Cross) + 4 (W) = 22MB.
        // We need bigger arena.
        // Let's optimize: Compute Cross into A_mat space if Batch <= D?
        // Batch=500, D=1024. A_mat is 2MB. Cross is 4MB. Can't fit.
        // We will just increase Arena to 32MB for demo.

        float* Cross = workspace_->allocate<float>(IN_DIM * IN_DIM);
        for(size_t r=0; r<IN_DIM; ++r) {
            for(size_t c=0; c<IN_DIM; ++c) {
                float sum = 0.0f;
                for(size_t k=0; k<batch_size; ++k) {
                    sum += A_mat[k*IN_DIM + r] * batch_X[k*IN_DIM + c];
                }
                Cross[r*IN_DIM + c] = sum;
            }
        }

        // W^T
        float* W_T = workspace_->allocate<float>(IN_DIM * IN_DIM);
        for(size_t r=0; r<IN_DIM; ++r) {
            for(size_t c=0; c<IN_DIM; ++c) {
                float sum = 0.0f;
                for(size_t k=0; k<IN_DIM; ++k) {
                    sum += Cov[r*IN_DIM + k] * Cross[k*IN_DIM + c];
                }
                W_T[r*IN_DIM + c] = sum;
            }
        }

        for(size_t i=0; i<IN_DIM * IN_DIM; ++i) {
            dec_readout.weights[i] = quant::APoT::quantize(W_T[i]);
        }
        std::cout << "Calibration Complete." << std::endl;
    }
};

} // namespace jules
} // namespace dreidel

// Define Global Agent (32MB)
static uint8_t ARENA_BUF[32 * 1024 * 1024];
static dreidel::core::Arena arena(ARENA_BUF, sizeof(ARENA_BUF));
static dreidel::jules::DeepJulesAutoencoder<1024, 64> agent(&arena);

int main() {
    std::cout << "=== Project Jules: Deep Autoencoder Demo ===" << std::endl;
    std::cout << "Fused Blocks: [Sparse || FWHT+SoftPerm] x 2 Layers" << std::endl;

    // 1. Generate Training Data
    size_t train_size = 500;
    std::cout << "Generating " << train_size << " wavelets..." << std::endl;

    dreidel::Tensor<float> train_data({train_size, 1024});
    dreidel::utils::generate_mixed_wavelets(train_data, train_size, 1024);
    std::vector<float> train_vec(train_data.data(), train_data.data() + train_data.size());

    // 2. Init & Calibrate
    agent.init();
    agent.calibrate(train_vec, train_size);

    // 3. Inference
    size_t val_size = 1;
    dreidel::Tensor<float> val_data({val_size, 1024});
    dreidel::utils::generate_mixed_wavelets(val_data, val_size, 1024);

    float output_buf[1024];
    auto start = std::chrono::high_resolution_clock::now();
    agent.step(val_data.data(), output_buf);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Inference Time: " << std::chrono::duration<double>(end - start).count() * 1000.0 << " ms" << std::endl;

    // 4. SVG Visualization
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
