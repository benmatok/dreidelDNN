#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdlib>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithNano.hpp"
#include "../include/dreidel/models/ZenithDiscriminator.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/io/Checkpoint.hpp"
#include <fstream>
#include <cstdio>
#include <filesystem>

using namespace dreidel;

// Helper to save tensor for Python LPIPS
void save_tensor_binary(const Tensor<float>& t, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(t.data()), t.size() * sizeof(float));
}

// Helper to load gradient from Python LPIPS
void load_tensor_binary(Tensor<float>& t, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (f.is_open()) {
        f.read(reinterpret_cast<char*>(t.data()), t.size() * sizeof(float));
    } else {
        std::cerr << "Failed to load gradient " << path << std::endl;
        t.fill(0);
    }
}

// Reusing wavelet generation from train_zenith_nano.cpp
void generate_wavelet_batch(Tensor<float>& data) {
    auto shape = data.shape();
    size_t batch = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> center_dist(0.3f, 0.7f);
    std::uniform_real_distribution<float> scale_dist(0.05f, 0.2f);

    data.fill(0.0f);
    float* ptr = data.data();

    for(size_t n=0; n<batch; ++n) {
        for(int k=0; k<4; ++k) {
            float cx = center_dist(gen) * W;
            float cy = center_dist(gen) * H;
            float s = scale_dist(gen) * W;

            float r = (float)gen() / gen.max();
            float g = (float)gen() / gen.max();
            float b = (float)gen() / gen.max();

            #pragma omp parallel for collapse(2)
            for(size_t y=0; y<H; ++y) {
                for(size_t x=0; x<W; ++x) {
                    float dx = x - cx;
                    float dy = y - cy;
                    float val = std::exp(-(dx*dx + dy*dy) / (2*s*s));
                    if (val > 0.01f) {
                        size_t idx = ((n*H + y)*W + x)*C;
                        ptr[idx + 0] += val * r;
                        ptr[idx + 1] += val * g;
                        ptr[idx + 2] += val * b;
                    }
                }
            }
        }
    }
}

// Function to load real data batch from file
bool load_real_batch(Tensor<float>& data, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open batch file " << filepath << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!file) {
        std::cerr << "Error: Could not read full batch from " << filepath << std::endl;
        return false;
    }
    return true;
}

int main() {
    std::cout << "=== Training ZenithNano: Wavelet Pretraining -> Real Data Finetuning ===" << std::endl;

    // Config
    size_t batch_size = 4;
    size_t H = 512, W = 512;
    float lr = 1e-4f;

    // Phase 1: Wavelet Pretraining
    size_t pretrain_epochs = 10000;

    // Phase 2: Real Data
    size_t real_epochs = 1000000;

    // Model (Generator)
    models::ZenithNano model(batch_size);
    optim::SimpleAdam<float> optimizer(lr);
    optimizer.add_parameters(model.parameters(), model.gradients());

    // Discriminator
    models::ZenithDiscriminator discriminator(batch_size);
    optim::SimpleAdam<float> opt_d(lr);
    opt_d.add_parameters(discriminator.parameters(), discriminator.gradients());

    // Checkpoints
    std::string ckpt_g = "checkpoint_G.bin";
    std::string ckpt_d = "checkpoint_D.bin";
    bool loaded = false;

    if (io::CheckpointManager::load(ckpt_g, model.parameters())) {
        std::cout << "Loaded Generator Checkpoint: " << ckpt_g << std::endl;
        loaded = true;
    }
    if (io::CheckpointManager::load(ckpt_d, discriminator.parameters())) {
        std::cout << "Loaded Discriminator Checkpoint: " << ckpt_d << std::endl;
    }

    Tensor<float> input({batch_size, H, W, 3});

    // ---------------------------------------------------------
    // PHASE 1: Wavelet Pretraining
    // ---------------------------------------------------------
    if (!loaded) {
        std::cout << "\n--- Phase 1: Wavelet Pretraining (" << pretrain_epochs << " epochs) ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        for(size_t epoch = 0; epoch < pretrain_epochs; ++epoch) {
            generate_wavelet_batch(input);

            Tensor<float> output = model.forward(input);

            // Loss (MAE)
            float loss = 0;
            size_t size = output.size();
            const float* out_ptr = output.data();
            const float* tgt_ptr = input.data();
            Tensor<float> grad_output(output.shape());
            float* go_ptr = grad_output.data();

            #pragma omp parallel for reduction(+:loss)
            for(size_t i=0; i<size; ++i) {
                float diff = out_ptr[i] - tgt_ptr[i];
                loss += std::abs(diff);
                float sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
                go_ptr[i] = sign / size;
            }
            loss /= size;

            optimizer.zero_grad();
            model.backward(grad_output);
            optimizer.step();

            if (epoch % 1000 == 0 || epoch == pretrain_epochs - 1) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                std::cout << "Pretrain Epoch " << epoch << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "s | Loss: " << std::setprecision(6) << loss << std::endl;
            }
        }

        // Save after pretraining
        io::CheckpointManager::save("checkpoint_pretrain_G.bin", model.parameters());
    } else {
        std::cout << "Skipping Pretraining (Checkpoint Loaded)" << std::endl;
    }

    // ---------------------------------------------------------
    // PHASE 2: Real Data Training
    // ---------------------------------------------------------
    std::cout << "\n--- Phase 2: Real Data Finetuning (" << real_epochs << " batches) ---" << std::endl;
    std::cout << "Downloading validation set..." << std::endl;

    // Download validation set once
    std::string val_cmd = "python3 tools/data_loader.py --batch-size " + std::to_string(batch_size) + " --output val_batch.bin --validation";
    int ret = std::system(val_cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to download validation set." << std::endl;
        return 1;
    }

    // Training loop
    for(size_t epoch = 0; epoch < real_epochs; ++epoch) {
        // Fetch new batch
        std::string train_cmd = "python3 tools/data_loader.py --batch-size " + std::to_string(batch_size) + " --output train_batch.bin";
        // To be less verbose, redirect stdout/stderr or just let it print
        // user wants to see "download small set for validation and train on new batches each time"
        // Let's print fetching status
        std::cout << "Fetching real batch " << epoch + 1 << "/" << real_epochs << "..." << std::endl;
        ret = std::system(train_cmd.c_str());
        if (ret != 0) {
             std::cerr << "Failed to download training batch. Skipping." << std::endl;
             continue;
        }

        if (!load_real_batch(input, "train_batch.bin")) {
            continue;
        }

        Tensor<float> output = model.forward(input);

        // ---------------------------------------------------------
        // GAN + LPIPS + L1 Step
        // ---------------------------------------------------------

        // 1. Compute LPIPS Gradients (via Python) - Only if enabled
        float lpips_val = 0.0f;
        Tensor<float> grad_lpips(output.shape());
        grad_lpips.fill(0.0f);

        if (w_lpips > 0.0f) {
            save_tensor_binary(output, "pred.bin");
            save_tensor_binary(input, "target.bin");

            std::string lpips_cmd = "python3 tools/compute_lpips.py --pred pred.bin --target target.bin --grad grad_lpips.bin --shape "
                                    + std::to_string(batch_size) + " " + std::to_string(H) + " " + std::to_string(W) + " 3 > lpips_loss.txt";
            int ret_lpips = std::system(lpips_cmd.c_str());

            if (ret_lpips == 0) {
                std::ifstream f("lpips_loss.txt");
                f >> lpips_val;
                load_tensor_binary(grad_lpips, "grad_lpips.bin");
            } else {
                 std::cerr << "LPIPS calculation failed. Ensure torch/lpips are installed or set w_lpips=0." << std::endl;
            }
        }

        // 2. Discriminator Update
        opt_d.zero_grad();

        // D(Real)
        Tensor<float> d_real = discriminator.forward(input); // Logits
        // Loss D_Real = (D(x) - 1)^2 (LSGAN)
        // dLoss/dD = 2*(D(x) - 1)
        Tensor<float> grad_d_real(d_real.shape());
        float loss_d_real = 0.0f;

        {
            float* g_ptr = grad_d_real.data();
            const float* d_ptr = d_real.data();
            size_t sz = d_real.size();
            #pragma omp parallel for reduction(+:loss_d_real)
            for(size_t i=0; i<sz; ++i) {
                float val = d_ptr[i];
                float diff = val - 1.0f;
                loss_d_real += diff * diff;
                g_ptr[i] = 2.0f * diff / sz; // Mean reduction
            }
            loss_d_real /= sz;
        }
        discriminator.backward(grad_d_real); // Accumulate grads for real

        // D(Fake)
        // Detach output (no backprop to Generator here)
        Tensor<float> d_fake = discriminator.forward(output);
        // Loss D_Fake = (D(G(z)))^2
        // dLoss/dD = 2*D(G(z))
        Tensor<float> grad_d_fake(d_fake.shape());
        float loss_d_fake = 0.0f;

        {
            float* g_ptr = grad_d_fake.data();
            const float* d_ptr = d_fake.data();
            size_t sz = d_fake.size();
            #pragma omp parallel for reduction(+:loss_d_fake)
            for(size_t i=0; i<sz; ++i) {
                float val = d_ptr[i];
                loss_d_fake += val * val;
                g_ptr[i] = 2.0f * val / sz;
            }
            loss_d_fake /= sz;
        }
        discriminator.backward(grad_d_fake); // Accumulate grads for fake

        opt_d.step(); // Update Discriminator

        // 3. Generator Update
        optimizer.zero_grad();

        // Re-forward D(Fake) for Generator loss?
        // Technically D weights changed, but usually we can reuse or just re-run.
        // Let's re-run D(Fake) with updated D? No, standard is simultaneous or alternating steps.
        // If alternating, we use the D state at this step.
        // We need gradients through D to G.

        // We need to re-forward D on output (with gradient tape for input of D)
        // dreidel::Tensor doesn't keep a tape. We need to call backward on D again
        // but this time propagate to input (which is output of G).

        // Wait, dreidel models (ZenithDiscriminator) accumulate gradients in their parameters.
        // If we call backward again, we corrupt D gradients?
        // We already stepped opt_d. So D gradients are zeroed effectively (or we should zero them if we don't want to update D again).
        // Yes, opt_d.zero_grad() was called before D update. Now we want G update.
        // D parameters are updated.
        // We don't want to update D parameters now. We just want dLoss/dInput.
        // But `discriminator.backward()` computes param grads AND input grads.
        // That's fine, we just ignore D param grads (don't call opt_d.step()).

        // Clear D grads just in case (though unused)
        // Actually, we don't have a manual zero_grad on model, only via optimizer.
        // But we won't step optimizer, so it's fine.

        // Forward D(G(z))
        Tensor<float> d_fake_g = discriminator.forward(output);

        // Loss G_GAN = (D(G(z)) - 1)^2 (Try to fool D to think it's real)
        Tensor<float> grad_gan_loss(d_fake_g.shape());
        float loss_g_gan = 0.0f;

        {
            float* g_ptr = grad_gan_loss.data();
            const float* d_ptr = d_fake_g.data();
            size_t sz = d_fake_g.size();
            #pragma omp parallel for reduction(+:loss_g_gan)
            for(size_t i=0; i<sz; ++i) {
                float val = d_ptr[i];
                float diff = val - 1.0f;
                loss_g_gan += diff * diff;
                g_ptr[i] = 2.0f * diff / sz;
            }
            loss_g_gan /= sz;
        }

        // Backprop through D to get grad w.r.t output (G output)
        Tensor<float> grad_from_d = discriminator.backward(grad_gan_loss);

        // Total Generator Gradient = L1_grad + LPIPS_grad + GAN_grad
        // Weights: L1=1.0, LPIPS=1.0, GAN=0.1 (Example)
        // NOTE: LPIPS requires PyTorch and is disabled by default to avoid heavy dependencies.
        // Set w_lpips > 0.0f and ensure requirements are installed to use it.
        float w_l1 = 1.0f;
        float w_lpips = 0.0f;
        float w_gan = 0.1f;

        Tensor<float> total_grad(output.shape());
        float* tot_ptr = total_grad.data();
        const float* lpips_ptr = grad_lpips.data();
        const float* gan_ptr = grad_from_d.data();
        const float* out_ptr = output.data();
        const float* tgt_ptr = input.data();
        size_t size = output.size();

        float loss_l1 = 0;

        #pragma omp parallel for reduction(+:loss_l1)
        for(size_t i=0; i<size; ++i) {
            float l1_diff = out_ptr[i] - tgt_ptr[i];
            loss_l1 += std::abs(l1_diff);
            float l1_grad = (l1_diff > 0) ? 1.0f : ((l1_diff < 0) ? -1.0f : 0.0f);
            l1_grad /= size;

            tot_ptr[i] = w_l1 * l1_grad + w_lpips * lpips_ptr[i] + w_gan * gan_ptr[i];
        }
        loss_l1 /= size;

        model.backward(total_grad);
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | L1: " << loss_l1
                    << " | LPIPS: " << lpips_val
                    << " | D_Real: " << loss_d_real
                    << " | D_Fake: " << loss_d_fake
                    << " | G_GAN: " << loss_g_gan << std::endl;
        }

        // Checkpointing every 1000 epochs
        if ((epoch + 1) % 1000 == 0) {
            io::CheckpointManager::save(ckpt_g, model.parameters());
            io::CheckpointManager::save(ckpt_d, discriminator.parameters());
        }

        // Validation every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            std::cout << "Validating..." << std::endl;
            if (load_real_batch(input, "val_batch.bin")) {
                 Tensor<float> val_out = model.forward(input);
                 float val_loss = 0;
                 const float* v_out = val_out.data();
                 const float* v_tgt = input.data();
                 #pragma omp parallel for reduction(+:val_loss)
                 for(size_t i=0; i<size; ++i) {
                     val_loss += std::abs(v_out[i] - v_tgt[i]);
                 }
                 val_loss /= size;
                 std::cout << ">>> Validation Loss: " << val_loss << std::endl;
            }
        }
    }

    // Final Save
    io::CheckpointManager::save(ckpt_g, model.parameters());
    io::CheckpointManager::save(ckpt_d, discriminator.parameters());

    std::cout << "Training Complete." << std::endl;
    return 0;
}
