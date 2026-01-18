#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <thread>
#include <fstream>
#include <string>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithHydra.hpp"
#include "../include/dreidel/models/ZenithDiscriminator.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/io/Checkpoint.hpp"
#include "../include/dreidel/transforms/Augmentations.hpp"

namespace fs = std::filesystem;
using namespace dreidel;

// Data Grazer Client
// Reads files from dataset/pool, loads them into a large buffer, and provides random crops.
class DataGrazer {
public:
    DataGrazer(const std::string& pool_dir) : pool_dir_(pool_dir) {}

    bool load_random_atlas(Tensor<float>& atlas) {
        // List files
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(pool_dir_)) {
            if (entry.path().extension() == ".bin") {
                files.push_back(entry.path().string());
            }
        }

        if (files.empty()) return false;

        // Pick random file
        static std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dis(0, files.size() - 1);
        std::string filepath = files[dis(gen)];

        // Load
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) return false;

        // Assume file is raw float32. Atlas size: 2048x2048x3
        // Check size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size != 2048*2048*3*sizeof(float)) {
            // Might be a partial file or different size, try to guess or skip
            // If it's 2048x2048, okay. If not, maybe just skip.
            // We'll enforce 2048x2048 in daemon.
            if (size == 0) return false;
        }

        file.read(reinterpret_cast<char*>(atlas.data()), size);
        return true;
    }

    // Deletes the file after use? No, "Grazing" implies we can reuse,
    // but the daemon overwrites or we just keep picking random ones.
    // The daemon maintains the pool. We just read.
    // To ensure freshness, maybe we delete old files?
    // The daemon logic was: "If count < 50, download more".
    // It doesn't delete automatically. The consumer (us) should consume.
    // Let's delete the file after loading to force the daemon to fetch new ones.
    bool consume_random_atlas(Tensor<float>& atlas) {
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(pool_dir_)) {
             if (entry.path().extension() == ".bin") {
                files.push_back(entry.path().string());
            }
        }
        if (files.empty()) return false;

        static std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dis(0, files.size() - 1);
        std::string filepath = files[dis(gen)];

        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) return false;

        file.read(reinterpret_cast<char*>(atlas.data()), atlas.size() * sizeof(float));
        file.close();

        // Delete file to signal consumption
        fs::remove(filepath);
        return true;
    }

private:
    std::string pool_dir_;
};

// Loss Helpers
float mse_loss(const Tensor<float>& pred, const Tensor<float>& target, Tensor<float>& grad, float scale=1.0f) {
    float loss = 0.0f;
    float* g = grad.data();
    const float* p = pred.data();
    const float* t = target.data();
    size_t size = pred.size();

    #pragma omp parallel for reduction(+:loss)
    for(size_t i=0; i<size; ++i) {
        float diff = p[i] - t[i];
        loss += diff * diff;
        g[i] += 2.0f * diff / size * scale; // Accumulate gradient
    }
    return loss / size;
}

float l1_loss(const Tensor<float>& pred, const Tensor<float>& target, Tensor<float>& grad, float scale=1.0f) {
    float loss = 0.0f;
    float* g = grad.data();
    const float* p = pred.data();
    const float* t = target.data();
    size_t size = pred.size();

    #pragma omp parallel for reduction(+:loss)
    for(size_t i=0; i<size; ++i) {
        float diff = p[i] - t[i];
        loss += std::abs(diff);
        float sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
        g[i] += sign / size * scale;
    }
    return loss / size;
}

int main(int argc, char** argv) {
    std::cout << "=== Zenith-Hydra: Robustness Pipeline ===" << std::endl;

    // Start Grazer Daemon
    // Check if running already? Just launch it in background.
    // Assuming python3 is available.
    std::string daemon_cmd = "python3 tools/grazer_daemon.py > grazer.log 2>&1 &";
    std::system(daemon_cmd.c_str());
    std::cout << "Launched Grazer Daemon." << std::endl;

    // Config
    size_t batch_size = 1;
    size_t epochs = 100000;

    models::ZenithHydra hydra(batch_size);
    models::ZenithDiscriminator discriminator(batch_size); // For Dreamer path

    optim::SimpleAdam<float> opt_hydra(1e-4f);
    opt_hydra.add_parameters(hydra.parameters(), hydra.gradients());

    optim::SimpleAdam<float> opt_disc(1e-4f);
    opt_disc.add_parameters(discriminator.parameters(), discriminator.gradients());

    // Tensors
    Tensor<float> atlas({1, 2048, 2048, 3});

    // View A (Target), View B (Robust/Input), View C (Chaos/Input)
    // Actually:
    // 1. Clean Path: Enc(Clean) -> Purist -> Clean. Target: Clean.
    // 2. Robust Path: Enc(Masked) -> Dreamer -> Clean. Target: Clean.
    // 3. Stable Path: Enc(Chaos) -> Purist -> Clean (or Chaos?).
    //    Plan: "Stable Path: Encoder(Noisy) -> Decoder_Purist -> Out_C".
    //    Is Target Clean or Noisy? "Goal: Fidelity". Usually Denoising Autoencoder target is Clean.
    //    So Target is always Clean (View A).
    //    Input varies: Clean (View A), Masked (View B), Chaos (View C).

    Tensor<float> view_a({batch_size, 512, 512, 3}); // Clean / Target
    Tensor<float> view_b({batch_size, 512, 512, 3}); // Masked
    Tensor<float> view_c({batch_size, 512, 512, 3}); // Chaos

    // Outputs
    Tensor<float> out_clean({batch_size, 512, 512, 3});
    Tensor<float> out_dream({batch_size, 512, 512, 3});
    Tensor<float> out_stable({batch_size, 512, 512, 3});

    // Gradients
    Tensor<float> grad_clean({batch_size, 512, 512, 3});
    Tensor<float> grad_dream({batch_size, 512, 512, 3});
    Tensor<float> grad_stable({batch_size, 512, 512, 3});

    DataGrazer grazer("dataset/pool");

    std::cout << "Waiting for data pool to populate..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // 1. Graze
        if (!grazer.consume_random_atlas(atlas)) {
            // Pool empty, wait
            std::cout << "Pool empty, waiting..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        // 2. Augment / Prepare Views
        // View A: Clean Random Crop
        transforms::Augmentations::RandomCrop(atlas, view_a, 512, 512);

        // View B: Masked (Copy A then Mask)
        transforms::Augmentations::Copy(view_a, view_b);
        transforms::Augmentations::PatchMask(view_b, 0.75f, 16);

        // View C: Chaos (Copy A then Augment)
        transforms::Augmentations::Copy(view_a, view_c);
        transforms::Augmentations::ApplyChaos(view_c, view_c); // In-place

        // 3. Forward Passes

        // Path 1: Clean (Purist)
        // Enc(View A) -> Purist -> Out Clean
        // Need to be careful with internal state if using one instance.
        // Hydra forward_purist calls encoder then decoder.
        // But encoder state is overwritten by subsequent calls?
        // ZenithEncoder stores state in s1_out_, etc.
        // If we call forward() 3 times sequentially, we overwrite the encoder state.
        // We cannot backprop through the first pass if we overwrote the state!
        // PROBLEM: We need to backprop through all 3 paths.
        // The current implementation of ZenithNano/Encoder stores activations in member variables.
        // It does NOT support concurrent forward passes or state retention for multiple passes in one batch unless batch size is larger.
        // BUT here we have 3 logical passes.
        // Solution:
        // Option A: Increase batch size to 3. Pack [A, B, C] into one batch.
        //   - Batch 0: View A
        //   - Batch 1: View B
        //   - Batch 2: View C
        //   Then we handle the outputs accordingly.
        //   Decoder Purist gets Batch 0 (Clean) and Batch 2 (Stable).
        //   Decoder Dreamer gets Batch 1 (Robust).
        //   This is elegant!
        // Option B: Run sequential, backprop immediately?
        //   - Forward Clean -> Backward Clean (accumulate grad)
        //   - Forward Robust -> Backward Robust
        //   - Forward Stable -> Backward Stable
        //   - Update.
        //   This works and uses less memory (Peak memory = 1 batch).
        //   "The Loss Equation (Auto-Balancing)" implies minimizing sum of losses.
        //   Gradients are additive. So sequential accumulation works perfectly.

        // Let's use Sequential Accumulation to save memory and complexity.

        opt_hydra.zero_grad();
        opt_disc.zero_grad();

        // Get Log Vars (Uncertainties)
        // tensor is [Clean, Robust, Stable]
        // We need the values to scale the loss and gradients.
        // Loss_i = (1 / 2*exp(s_i)) * L_i + 0.5 * s_i
        // dLoss_total / dL_i = 1 / 2*exp(s_i)
        // dLoss_total / ds_i = -0.5 * L_i * exp(-s_i) + 0.5

        // We need to fetch log_vars from GPU/Tensor? It's CPU.
        float* log_vars_ptr = hydra.parameters().back()->data(); // Assuming last param is log_vars
        float s_clean = log_vars_ptr[0];
        float s_robust = log_vars_ptr[1];
        float s_stable = log_vars_ptr[2];

        // Pre-compute scalers (1 / (2 * exp(s)))
        float scale_clean = 0.5f * std::exp(-s_clean);
        float scale_robust = 0.5f * std::exp(-s_robust);
        float scale_stable = 0.5f * std::exp(-s_stable);

        // Gradient for log_vars (accumulated manually or via backprop?)
        // Hydra parameters() includes log_vars, so optimizer updates it.
        // We need to populate hydra.gradients().back() with dLoss/ds.
        float* log_vars_grad_ptr = hydra.gradients().back()->data();
        // Initialize to 0.5 (the regularization term)
        log_vars_grad_ptr[0] = 0.5f;
        log_vars_grad_ptr[1] = 0.5f;
        log_vars_grad_ptr[2] = 0.5f;

        // --- Path 1: Clean (Purist) ---
        // Loss: LPIPS (or L1 as proxy)
        out_clean = hydra.forward_purist(view_a);
        grad_clean.fill(0.0f);
        // Using L1 for now as proxy for LPIPS (User constraint).
        // Ideally we'd use LPIPS if available.
        float l_clean = l1_loss(out_clean, view_a, grad_clean, scale_clean);
        // Backprop
        hydra.backward_purist(grad_clean);
        // Update log_var grad
        log_vars_grad_ptr[0] -= 0.5f * l_clean * std::exp(-s_clean); // Wait, use original loss L before scaling?
        // Formula: Loss = exp(-s)*L + s
        // d/ds = -exp(-s)*L + 1. (My formula above had 0.5, depends on definition).
        // Plan says: "1/(2 sigma^2) L + log sigma".
        // Let s = log(sigma^2). sigma^2 = exp(s). 2*sigma^2 = 2*exp(s).
        // Loss = exp(-s)/2 * L + 0.5 * s.
        // d/ds = -0.5 * exp(-s) * L + 0.5.
        // Yes.
        // Wait, l_clean computed by l1_loss is already scaled?
        // l1_loss returns raw L? No, I passed `scale` to accumulate gradient.
        // But the returned float loss should be raw L for the s-gradient calculation.
        // My l1_loss function returns averaged L (unscaled? No, checking code).
        // It returns `loss / size`. It does not apply scale to the returned float, only to the gradient tensor.
        // Correct.
        log_vars_grad_ptr[0] += -0.5f * l_clean * std::exp(-s_clean); // Add to the 0.5 initial


        // --- Path 2: Robust (Dreamer) ---
        // Loss: GAN
        // Input: view_b (Masked). Target: view_a (Clean).
        // This is Inpainting.
        out_dream = hydra.forward_dreamer(view_b);
        grad_dream.fill(0.0f);

        // GAN Loss
        // D update
        // Real: view_a
        Tensor<float> d_real = discriminator.forward(view_a);
        Tensor<float> g_d_real(d_real.shape());
        // Loss D_Real = (D(x)-1)^2
        // d/dD = 2(D-1)
        float loss_d_real = 0.0f;
        // ... (Implement D loss calc and backprop) ...
        // Note: D is not multi-headed, it's just one D?
        // "Head 2: ... Loss: GAN".
        // Usually we need a D to judge the inpainting.
        // We perform D update here.
        {
             float* g = g_d_real.data();
             const float* d = d_real.data();
             #pragma omp parallel for reduction(+:loss_d_real)
             for(size_t i=0; i<d_real.size(); ++i) {
                 float val = d[i];
                 loss_d_real += (val - 1.0f) * (val - 1.0f);
                 g[i] = 2.0f * (val - 1.0f) / d_real.size();
             }
             loss_d_real /= d_real.size();
        }
        discriminator.backward(g_d_real);

        // Fake: out_dream
        Tensor<float> d_fake = discriminator.forward(out_dream); // Detach implicit
        Tensor<float> g_d_fake(d_fake.shape());
        float loss_d_fake = 0.0f;
        {
             float* g = g_d_fake.data();
             const float* d = d_fake.data();
             #pragma omp parallel for reduction(+:loss_d_fake)
             for(size_t i=0; i<d_fake.size(); ++i) {
                 float val = d[i];
                 loss_d_fake += val * val;
                 g[i] = 2.0f * val / d_fake.size();
             }
             loss_d_fake /= d_fake.size();
        }
        discriminator.backward(g_d_fake);
        opt_disc.step(); // Update D

        // G Update (Generator/Hydra)
        // Forward D again on G output
        // We need to re-forward Dreamer? No, we have out_dream.
        // Forward D(out_dream)
        Tensor<float> d_fake_g = discriminator.forward(out_dream);
        Tensor<float> g_gan(d_fake_g.shape());
        float l_gan = 0.0f;
        {
             float* g = g_gan.data();
             const float* d = d_fake_g.data();
             #pragma omp parallel for reduction(+:l_gan)
             for(size_t i=0; i<d_fake_g.size(); ++i) {
                 float val = d[i];
                 l_gan += (val - 1.0f) * (val - 1.0f);
                 g[i] = 2.0f * (val - 1.0f) / d_fake_g.size();
             }
             l_gan /= d_fake_g.size();
        }
        // Backprop D to get grad w.r.t input (out_dream)
        // We use scale_robust for this gradient
        Tensor<float> grad_from_d = discriminator.backward(g_gan); // This returns dLoss/dInput

        // Scale gradient by uncertainty
        float* gd_ptr = grad_from_d.data();
        #pragma omp parallel for
        for(size_t i=0; i<grad_from_d.size(); ++i) {
            gd_ptr[i] *= scale_robust;
        }

        // Add L1/MSE reconstruction loss to Dreamer as well?
        // Plan says "Loss: GAN". Usually also needs reconstruction loss to ground it.
        // Assuming pure GAN might be unstable. But following plan: "Loss: GAN".
        // I'll add a small L1 weight just in case, or stick to plan.
        // "Head 2... Goal: Plausible Hallucination".
        // If I only use GAN, it might hallucinate *anything*.
        // It should match the unmasked parts!
        // So implicitly it should reconstruct unmasked parts.
        // MAE usually computes loss only on masked patches.
        // I will add L1 loss on the WHOLE image or just masked?
        // Plan doesn't specify. I'll stick to GAN loss as the primary "Robust Path" driver.
        // But usually "Structure Recovery" implies matching context.
        // Let's rely on GAN for now as per plan "Loss: GAN".

        hydra.backward_dreamer(grad_from_d);
        log_vars_grad_ptr[1] += -0.5f * l_gan * std::exp(-s_robust);


        // --- Path 3: Stable (Purist) ---
        // Loss: MSE
        // Input: view_c (Chaos/Noisy). Target: view_a (Clean).
        out_stable = hydra.forward_purist(view_c);
        grad_stable.fill(0.0f);
        float l_mse = mse_loss(out_stable, view_a, grad_stable, scale_stable);
        hydra.backward_purist(grad_stable);
        log_vars_grad_ptr[2] += -0.5f * l_mse * std::exp(-s_stable);

        // Update Hydra
        opt_hydra.step();

        // Logging
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Clean(L1): " << l_clean << " (s=" << s_clean << ")"
                      << " | Robust(GAN): " << l_gan << " (s=" << s_robust << ")"
                      << " | Stable(MSE): " << l_mse << " (s=" << s_stable << ")"
                      << std::endl;
        }

        // Checkpoint
        if (epoch > 0 && epoch % 1000 == 0) {
            io::CheckpointManager::save("hydra_ckpt.bin", hydra.parameters());
        }
    }

    std::cout << "Training Complete." << std::endl;
    return 0;
}
