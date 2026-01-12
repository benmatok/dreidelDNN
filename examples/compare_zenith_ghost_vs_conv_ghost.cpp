#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>

#include "dreidel/core/Tensor.hpp"
#include "dreidel/models/ZenithGhostAE.hpp"
#include "dreidel/models/ConvGhostAE.hpp"
#include "dreidel/optim/SimpleAdam.hpp"
#include "dreidel/data/WaveletGenerator2D.hpp"

using namespace dreidel;

// Energy Conservation Loss (L_energy)
// Enforce stddev of Ghost Pred to match Target
// L = ReLU(1.0 - std(Pred)/std(Target))
template <typename T>
T compute_energy_loss(const Tensor<T>& pred, const Tensor<T>& target) {
    // 1. Compute Stds
    auto compute_std = [](const Tensor<T>& t) {
        T mean = 0;
        const T* d = t.data();
        size_t n = t.size();
        for(size_t i=0; i<n; ++i) mean += d[i];
        mean /= n;
        T var = 0;
        for(size_t i=0; i<n; ++i) var += (d[i] - mean)*(d[i] - mean);
        return std::sqrt(var / n);
    };

    T s_p = compute_std(pred);
    T s_t = compute_std(target);

    T ratio = s_p / (s_t + 1e-6f);
    // We want ratio >= 1.0 (actually approx 1.0, but usually collapse -> 0)
    // Loss = ReLU(1.0 - ratio)
    if (ratio < 1.0f) return (1.0f - ratio);
    return 0.0f;
}

template <typename T>
void train_model(layers::Layer<T>* model, const std::string& model_name, size_t epochs, size_t batch_size = 4) {
    std::cout << "\n=== Training " << model_name << " ===" << std::endl;

    optim::SimpleAdam<T> optimizer(1e-5);
    optimizer.add_parameters(model->parameters(), model->gradients());
    // Lower learning rate for stable convergence

    data::WaveletGenerator2D<T> generator(128, 128, 3);

    // Fixed validation set
    auto val_batch = generator.generate_batch(batch_size);

    auto start_time = std::chrono::high_resolution_clock::now();

    for(size_t epoch=0; epoch<epochs; ++epoch) {
        auto batch = generator.generate_batch(batch_size);

        // Forward
        // We use forward_train if available, but Layer interface only exposes forward.
        // We need to cast to specific type.

        Tensor<T> recon;
        std::vector<Tensor<T>> ghost_preds;
        std::vector<Tensor<T>> targets;

        if (auto* z_model = dynamic_cast<models::ZenithGhostAE<T>*>(model)) {
            auto out = z_model->forward_train(batch);
            recon = out.reconstruction;
            ghost_preds = out.ghost_preds;
            targets = out.encoder_targets;
        } else if (auto* c_model = dynamic_cast<models::ConvGhostAE<T>*>(model)) {
            auto out = c_model->forward_train(batch);
            recon = out.reconstruction;
            ghost_preds = out.ghost_preds;
            targets = out.encoder_targets;
        }

        // Loss Calculation
        // 1. Reconstruction Loss (MSE)
        T recon_loss = 0;
        size_t n_elem = recon.size();
        Tensor<T> d_recon(recon.shape());
        const T* r_ptr = recon.data();
        const T* t_ptr = batch.data();
        T* dr_ptr = d_recon.data();

        #pragma omp parallel for reduction(+:recon_loss)
        for(size_t i=0; i<n_elem; ++i) {
            T diff = r_ptr[i] - t_ptr[i];
            recon_loss += diff * diff;
            dr_ptr[i] = 2.0f * diff / n_elem;
        }
        recon_loss /= n_elem;

        // 2. Feature Consistency Loss (MSE + Energy)
        T feat_loss = 0;
        std::vector<Tensor<T>> d_ghosts;

        for(size_t i=0; i<ghost_preds.size(); ++i) {
            Tensor<T>& pred = ghost_preds[i];
            Tensor<T>& targ = targets[i]; // Treat as const target (detached)

            size_t sz = pred.size();
            Tensor<T> d_g(pred.shape());
            T* dg_ptr = d_g.data();
            const T* p_ptr = pred.data();
            const T* tg_ptr = targ.data();

            T l_mse = 0;
            #pragma omp parallel for reduction(+:l_mse)
            for(size_t k=0; k<sz; ++k) {
                T diff = p_ptr[k] - tg_ptr[k];
                l_mse += diff * diff;
                dg_ptr[k] = 2.0f * diff / sz;
            }
            l_mse /= sz;

            // Energy Loss (Naive gradients for std)
            // L = ReLU(1 - sigma_p / sigma_t)
            // dL/dp = ... complicated. We use a simple proxy:
            // if std(p) < std(t), push p away from mean.
            // dL/dp += -0.1 * (p - mean)

            T energy_l = compute_energy_loss(pred, targ);
            if (energy_l > 0) {
                 // Backward proxy
                 T mean = 0;
                 for(size_t k=0; k<sz; ++k) mean += p_ptr[k];
                 mean /= sz;

                 for(size_t k=0; k<sz; ++k) {
                     dg_ptr[k] -= 0.01f * (p_ptr[k] - mean); // Push energy up
                 }
            }

            feat_loss += l_mse + energy_l;
            d_ghosts.push_back(d_g);
        }

        // Backward
        optimizer.zero_grad();

        if (auto* z_model = dynamic_cast<models::ZenithGhostAE<T>*>(model)) {
            z_model->backward_train(d_recon, d_ghosts);
        } else if (auto* c_model = dynamic_cast<models::ConvGhostAE<T>*>(model)) {
            c_model->backward_train(d_recon, d_ghosts);
        }

        optimizer.step();

        std::cout << "Epoch " << epoch << ": Recon Loss=" << recon_loss
                  << " Feat Loss=" << feat_loss << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Training Time: " << diff.count() << " s" << std::endl;
    std::cout << "Samples/Sec: " << (epochs * batch_size) / diff.count() << std::endl;

    // Final Validation
    Tensor<T> val_recon;
    if (auto* z_model = dynamic_cast<models::ZenithGhostAE<T>*>(model)) {
        val_recon = z_model->forward(val_batch);
    } else if (auto* c_model = dynamic_cast<models::ConvGhostAE<T>*>(model)) {
        val_recon = c_model->forward(val_batch);
    }

    T val_mse = 0;
    const T* v_ptr = val_recon.data();
    const T* gt_ptr = val_batch.data();
    for(size_t i=0; i<val_recon.size(); ++i) {
        T d = v_ptr[i] - gt_ptr[i];
        val_mse += d*d;
    }
    val_mse /= val_recon.size();
    std::cout << "Final Validation MSE: " << val_mse << std::endl;
}

int main() {
    // Set threads
    omp_set_num_threads(4);
    std::cout << "Running Benchmark on 128x128 Wavelet Data" << std::endl;

    size_t epochs = 10;

    // 1. Train ZenithGhostAE
    {
        models::ZenithGhostAE<float> zenith_model;
        train_model(&zenith_model, "ZenithGhostAE", epochs);
    }

    // 2. Train ConvGhostAE
    {
        models::ConvGhostAE<float> conv_model;
        train_model(&conv_model, "ConvGhostAE", epochs);
    }

    return 0;
}
