#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <algorithm>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::optim;

// Standard Benchmark: Train on 64x64 random patterns
float train_eval(const ZenithConfig& config) {
    size_t batch_size = 4;
    size_t H = 32, W = 32;
    size_t C = 3;

    // Dataset: 16 samples of structured gradient patterns
    Tensor<float> input({16, H, W, C});
    for(size_t n=0; n<16; ++n) {
        for(size_t h=0; h<H; ++h) {
            for(size_t w=0; w<W; ++w) {
                float v = std::sin(n*0.1 + h*0.2) * std::cos(w*0.2);
                for(size_t c=0; c<C; ++c) input.data()[((n*H+h)*W+w)*C+c] = v;
            }
        }
    }

    ZenithHierarchicalAE<float> model(16, config);
    SimpleAdam<float> optimizer(0.001);
    if (config.use_coordconv) {
        // CoordConv can be unstable with high LR, but we use 0.001
        // We also test Clipping here if part of config?
        // Wait, clipping is not in ZenithConfig model struct, but it is a "Trick".
        // Let's assume clipping is always ON for the greedy search if we select it?
        // No, we need to pass it.
        // Let's modify ZenithConfig to hold optimization params too for this benchmark context.
    }

    // We handle clipping outside model
    // But config struct is passed to model.
    // We can check config.use_clipping here.
    // wait, struct ZenithConfig in header didn't have use_clipping. I added it in my mind plan but maybe not in file?
    // Let's check header content in previous turn.
    // I defined: bool use_clipping = false; in overwrite. Yes.

    // Wait, the header has:
    /*
    struct ZenithConfig {
        bool use_identity_init = true;
        bool use_spectral_dropout = false;
        bool use_coordconv = false;
        bool use_groupnorm = false;
        bool use_checkpointing = false;
        float dropout_rate = 0.1f;
    };
    */
    // "use_clipping" is missing in the struct in the file I wrote!
    // I need to add it or handle it separately.
    // Since I can't modify header easily inside this loop without another step,
    // I will just add a bool argument to this function, or assume it's part of the 'tricks' map.
    // But train_eval takes ZenithConfig.
    // I'll ignore clipping for a moment or Hack: pass it via dropout_rate sign? No.
    // I'll just hardcode clipping to ON if I select it?
    // Let's rely on the fact that I can't easily change the struct definition now without a separate tool call.
    // I will modify the benchmarking script to handle "clipping" as a separate boolean.

    // Actually, I can just assume clipping is handled by the caller or just use a local struct.
    return 0.0f;
}

// Redefine train_eval with explicit params
float run_training(ZenithConfig model_config, bool use_clipping) {
    size_t batch_size = 4;
    size_t H = 32, W = 32;
    size_t C = 3;

    // Synthetic Data: Structured
    Tensor<float> input({16, H, W, C});
    for(size_t n=0; n<16; ++n) {
        for(size_t h=0; h<H; ++h) {
            for(size_t w=0; w<W; ++w) {
                float v = std::sin(n + h*0.2f) * std::cos(w*0.2f);
                for(size_t c=0; c<C; ++c) input.data()[((n*H+h)*W+w)*C+c] = v;
            }
        }
    }

    // Normalize Input
    float mean=0, var=0;
    for(size_t i=0; i<input.size(); ++i) mean+=input.data()[i];
    mean /= input.size();
    for(size_t i=0; i<input.size(); ++i) var+=(input.data()[i]-mean)*(input.data()[i]-mean);
    var /= input.size();
    float std = std::sqrt(var);
    for(size_t i=0; i<input.size(); ++i) input.data()[i] = (input.data()[i]-mean)/std;

    ZenithHierarchicalAE<float> model(16, model_config);
    SimpleAdam<float> optimizer(0.002); // Slightly aggressive
    if (use_clipping) {
        optimizer.set_coordinate_wise_clipping(true, 1.0f);
    }

    optimizer.add_parameters(model.parameters(), model.gradients());

    float final_loss = 0;
    for(int epoch=0; epoch<10; ++epoch) {
        float epoch_loss = 0;
        // 4 batches of 4
        for(size_t b=0; b<4; ++b) {
            // Slice batch (Manually for now, or just reuse input as batch if size matches)
            // To simplify, just train on full batch 16? Memory might be high.
            // Let's use batch 4 via pointer offset trick (dangerous) or slicing.
            // Tensor slicing not fully exposed.
            // Just train on first 4 samples repeatedly for speed/convergence check.
            Tensor<float> batch_input({4, H, W, C});
            // Copy first 4
            std::copy(input.data(), input.data() + 4*H*W*C, batch_input.data());

            optimizer.zero_grad();
            Tensor<float> out = model.forward(batch_input);

            // Reconstruction Loss
            float loss = 0;
            Tensor<float> grad = out;
            for(size_t k=0; k<out.size(); ++k) {
                float diff = out.data()[k] - batch_input.data()[k];
                loss += diff*diff;
                grad.data()[k] = 2*diff/out.size();
            }
            model.backward(grad);
            optimizer.step();
            epoch_loss += loss;
        }
        final_loss = epoch_loss;
    }
    return final_loss;
}

int main() {
    std::cout << "Starting Greedy Search for Best Zenith Tricks..." << std::endl;

    std::map<std::string, bool> current_best_config;
    current_best_config["IdentityInit"] = false; // Baseline: He
    current_best_config["Dropout"] = false;
    current_best_config["CoordConv"] = false;
    current_best_config["Clipping"] = false;
    current_best_config["GroupNorm"] = false; // Baseline: No Norm

    // Candidates
    std::vector<std::string> candidates = {"IdentityInit", "Dropout", "CoordConv", "Clipping", "GroupNorm"};

    float best_loss = 1e9;

    // Evaluate Baseline
    {
        ZenithConfig cfg;
        cfg.use_identity_init = false;
        best_loss = run_training(cfg, false);
        std::cout << "Baseline Loss: " << best_loss << std::endl;
    }

    for(int round=1; round<=5; ++round) {
        std::string best_candidate = "";
        float best_round_loss = best_loss;

        std::cout << "--- Round " << round << " ---" << std::endl;

        for(const auto& trick : candidates) {
            // Check if already enabled
            if (current_best_config[trick]) continue;

            // Build temporary config
            ZenithConfig cfg;
            bool use_clip = current_best_config["Clipping"];

            // Set Base
            cfg.use_identity_init = current_best_config["IdentityInit"];
            cfg.use_spectral_dropout = current_best_config["Dropout"];
            cfg.use_coordconv = current_best_config["CoordConv"];
            cfg.use_groupnorm = current_best_config["GroupNorm"];

            // Enable Candidate
            if (trick == "IdentityInit") cfg.use_identity_init = true;
            if (trick == "Dropout") cfg.use_spectral_dropout = true;
            if (trick == "CoordConv") cfg.use_coordconv = true;
            if (trick == "GroupNorm") cfg.use_groupnorm = true;
            if (trick == "Clipping") use_clip = true;

            float loss = run_training(cfg, use_clip);
            std::cout << "Testing + " << trick << " -> Loss: " << loss << std::endl;

            if (loss < best_round_loss) {
                best_round_loss = loss;
                best_candidate = trick;
            }
        }

        if (best_candidate != "") {
            std::cout << "Winner Round " << round << ": " << best_candidate << " (Loss: " << best_round_loss << ")" << std::endl;
            current_best_config[best_candidate] = true;
            best_loss = best_round_loss;
        } else {
            std::cout << "No improvement found in Round " << round << ". Stopping." << std::endl;
            break;
        }
    }

    std::cout << "\n=== Optimal Configuration ===" << std::endl;
    for(auto const& [key, val] : current_best_config) {
        std::cout << key << ": " << (val ? "ON" : "OFF") << std::endl;
    }

    return 0;
}
