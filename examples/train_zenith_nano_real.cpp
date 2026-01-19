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
#include <thread>
#include <mutex>
#include <atomic>

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

    // Clamp to [0, 1] to match real data distribution
    size_t sz = data.size();
    float* d_ptr = data.data();
    #pragma omp parallel for
    for(size_t i=0; i<sz; ++i) {
        if(d_ptr[i] > 1.0f) d_ptr[i] = 1.0f;
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

// Augmentation: Horizontal Flip + Histogram Stretch
void augment_batch(Tensor<float>& batch) {
    size_t N = batch.shape()[0];
    size_t H = batch.shape()[1];
    size_t W = batch.shape()[2];
    size_t C = batch.shape()[3];

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // For each image in batch
    for(size_t n=0; n<N; ++n) {
        // 1. Horizontal Flip
        if(dist(gen) > 0.5f) {
             #pragma omp parallel for
             for(size_t y=0; y<H; ++y) {
                 for(size_t x=0; x<W/2; ++x) {
                     for(size_t c=0; c<C; ++c) {
                         size_t idx1 = ((n*H + y)*W + x)*C + c;
                         size_t idx2 = ((n*H + y)*W + (W - 1 - x))*C + c;
                         std::swap(batch.data()[idx1], batch.data()[idx2]);
                     }
                 }
             }
        }

        // 2. Histogram Stretch (Contrast)
        // Find min and max for this image
        float min_val = 1.0f;
        float max_val = 0.0f;
        size_t offset = n * H * W * C;
        float* img_ptr = batch.data() + offset;
        size_t img_size = H * W * C;

        for(size_t i=0; i<img_size; ++i) {
            float val = img_ptr[i];
            if(val < min_val) min_val = val;
            if(val > max_val) max_val = val;
        }

        // If range is decent, stretch to [0, 1]
        if(max_val > min_val + 0.01f) {
            float range = max_val - min_val;
            #pragma omp parallel for
            for(size_t i=0; i<img_size; ++i) {
                img_ptr[i] = (img_ptr[i] - min_val) / range;
            }
        }
    }
}

// Threaded Data Buffer
class AsyncBuffer {
public:
    AsyncBuffer(size_t capacity, size_t batch_size)
        : capacity_(capacity), batch_size_(batch_size), running_(true)
    {
        // Initialize buffer with empty tensors
        for(size_t i=0; i<capacity_; ++i) {
             // 512x512x3
             buffer_.emplace_back(std::vector<size_t>{batch_size_, 512, 512, 3});
             buffer_.back().fill(0.0f);
             valid_.push_back(false);
        }
    }

    ~AsyncBuffer() {
        stop();
    }

    void start(const std::string& script_cmd) {
        worker_ = std::thread([this, script_cmd](){
             while(running_) {
                 // Download to temp file
                 std::string temp_file = "temp_load.bin";
                 // Using a random temp file or just one? One is fine if only 1 producer.

                 // Call Python script
                 std::string cmd = script_cmd + " --output " + temp_file + " > /dev/null 2>&1";
                 int ret = std::system(cmd.c_str());

                 // Rate limit: Enforce a delay to respect "50 requests / 90 seconds"
                 // 90s / 50 requests = 1.8s per request. We use 2.0s to be safe.
                 std::this_thread::sleep_for(std::chrono::seconds(2));

                 if(ret != 0) {
                     std::this_thread::sleep_for(std::chrono::milliseconds(100));
                     continue;
                 }

                 Tensor<float> t({batch_size_, 512, 512, 3});
                 if(!load_real_batch(t, temp_file)) continue;

                 // Update buffer
                 {
                     std::lock_guard<std::mutex> lock(mtx_);
                     // Use simple thread-safe random logic or just round robin?
                     // Random replacement is requested.
                     // Use a simple static thread_local generator
                     static thread_local std::mt19937 gen(std::random_device{}());
                     std::uniform_int_distribution<int> dist(0, capacity_ - 1);
                     int idx = dist(gen);

                     buffer_[idx].copy_from(t); // Deep copy
                     valid_[idx] = true;

                     // Optional: Print status periodically?
                 }
             }
        });
    }

    void stop() {
        running_ = false;
        if(worker_.joinable()) worker_.join();
    }

    // Copy a random valid batch to out
    bool get_batch(Tensor<float>& out) {
        std::lock_guard<std::mutex> lock(mtx_);

        // Find valid indices
        std::vector<int> indices;
        for(size_t i=0; i<capacity_; ++i) {
            if(valid_[i]) indices.push_back(i);
        }

        if(indices.empty()) return false;

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, indices.size() - 1);
        int idx = indices[dist(gen)];

        out.copy_from(buffer_[idx]);
        return true;
    }

    bool is_ready() {
        std::lock_guard<std::mutex> lock(mtx_);
        for(bool v : valid_) if(v) return true;
        return false;
    }

private:
    size_t capacity_;
    size_t batch_size_;
    std::vector<Tensor<float>> buffer_;
    std::vector<bool> valid_;
    std::mutex mtx_;
    std::atomic<bool> running_;
    std::thread worker_;
};

int main(int argc, char** argv) {
    std::cout << "=== Training ZenithNano: Wavelet Pretraining -> Real Data Finetuning ===" << std::endl;

    // Config
    size_t batch_size = 1;
    size_t H = 512, W = 512;
    float lr = 1e-4f;

    // Phase 1: Wavelet Pretraining
    size_t pretrain_epochs = 10000;

    // Phase 2: Real Data
    size_t real_epochs = 1000000;

    // Parse CLI args
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--skip-pretrain") {
            pretrain_epochs = 0;
            std::cout << "Config: Skipping Pretraining phase." << std::endl;
        }
    }

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

    // Start Data Producer
    std::cout << "Starting Data Producer Thread..." << std::endl;
    AsyncBuffer data_buffer(50, batch_size); // 50 batches buffer
    std::string train_cmd = "python3 tools/data_loader.py --batch-size " + std::to_string(batch_size);
    data_buffer.start(train_cmd);

    // Wait for buffer to have at least one batch
    std::cout << "Waiting for data buffer..." << std::endl;
    while(!data_buffer.is_ready()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Training loop
    for(size_t epoch = 0; epoch < real_epochs; ++epoch) {
        // Fetch from buffer
        if (!data_buffer.get_batch(input)) {
             // Should not happen if we waited
             std::this_thread::sleep_for(std::chrono::milliseconds(10));
             continue;
        }

        // Augment
        augment_batch(input);

        Tensor<float> output = model.forward(input);

        // ---------------------------------------------------------
        // GAN + LPIPS + L1 Step
        // ---------------------------------------------------------

        // Total Generator Gradient = L1_grad + LPIPS_grad + GAN_grad
        // Weights: L1=1.0, LPIPS=1.0, GAN=0.1 (Example)
        // NOTE: LPIPS requires PyTorch and is disabled by default to avoid heavy dependencies.
        // Set w_lpips > 0.0f and ensure requirements are installed to use it.
        float w_l1 = 1.0f;
        float w_lpips = 0.0f;
        float w_gan = 0.1f;

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
            // std::cout << "Validating..." << std::endl;
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

    // Stop Producer
    data_buffer.stop();

    // Final Save
    io::CheckpointManager::save(ckpt_g, model.parameters());
    io::CheckpointManager::save(ckpt_d, discriminator.parameters());

    std::cout << "Training Complete." << std::endl;
    return 0;
}
