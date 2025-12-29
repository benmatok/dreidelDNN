#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>

// Include Dreidel headers
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/Dense.hpp"
#include "../include/dreidel/layers/Conv2D.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/layers/Quantization.hpp" // Pack/Unpack

using namespace dreidel;

// Helpers
template <typename T>
class AvgPool2D : public layers::Layer<T> {
public:
    AvgPool2D(size_t stride) : stride_(stride) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        input_shape_ = input.shape();
        size_t N = input_shape_[0]; size_t H = input_shape_[1]; size_t W = input_shape_[2]; size_t C = input_shape_[3];
        size_t H_out = H / stride_; size_t W_out = W / stride_;
        Tensor<T> output({N, H_out, W_out, C});
        output.fill(0);
        T* out_ptr = output.data(); const T* in_ptr = input.data();
        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    for(size_t c=0; c<C; ++c) {
                        T sum = 0;
                        for(size_t dy=0; dy<stride_; ++dy) {
                            for(size_t dx=0; dx<stride_; ++dx) {
                                sum += in_ptr[((n*H + h*stride_+dy)*W + w*stride_+dx)*C + c];
                            }
                        }
                        out_ptr[((n*H_out + h)*W_out + w)*C + c] = sum / (stride_ * stride_);
                    }
                }
            }
        }
        return output;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); } // Dummy
    std::string name() const override { return "AvgPool2D"; }
private:
    size_t stride_;
    std::vector<size_t> input_shape_;
};

template <typename T>
class Upscale2D : public layers::Layer<T> {
public:
    Upscale2D(size_t scale) : scale_(scale) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        input_shape_ = input.shape();
        size_t N = input_shape_[0]; size_t H = input_shape_[1]; size_t W = input_shape_[2]; size_t C = input_shape_[3];
        size_t H_out = H * scale_; size_t W_out = W * scale_;
        Tensor<T> output({N, H_out, W_out, C});
        T* out_ptr = output.data(); const T* in_ptr = input.data();
        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_; size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        out_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = in_ptr[((n*H + h_in)*W + w_in)*C + c];
                    }
                }
            }
        }
        return output;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); }
    std::string name() const override { return "Upscale2D"; }
private:
    size_t scale_;
    std::vector<size_t> input_shape_;
};

template <typename T>
class Flatten : public layers::Layer<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        input_shape_ = input.shape();
        size_t batch = input_shape_[0];
        size_t dim = 1;
        for(size_t i=1; i<input_shape_.size(); ++i) dim *= input_shape_[i];
        Tensor<T> output({batch, dim});
        const T* in_ptr = input.data(); T* out_ptr = output.data();
        std::copy(in_ptr, in_ptr + output.size(), out_ptr);
        return output;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); }
    std::string name() const override { return "Flatten"; }
private:
    std::vector<size_t> input_shape_;
};

template <typename T>
class Reshape : public layers::Layer<T> {
public:
    Reshape(std::vector<size_t> target_shape_suffix) : target_suffix_(target_shape_suffix) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        input_shape_ = input.shape();
        size_t batch = input_shape_[0];
        std::vector<size_t> new_shape = {batch};
        new_shape.insert(new_shape.end(), target_suffix_.begin(), target_suffix_.end());
        Tensor<T> output(new_shape);
        const T* in_ptr = input.data(); T* out_ptr = output.data();
        std::copy(in_ptr, in_ptr + output.size(), out_ptr);
        return output;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); }
    std::string name() const override { return "Reshape"; }
private:
    std::vector<size_t> input_shape_;
    std::vector<size_t> target_suffix_;
};

// 2D Wavelet Generator (Restored Rich Families)
template <typename T>
void generate_wavelet_images(Tensor<T>& data) {
    auto shape = data.shape();
    size_t batch = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    // Use fixed seed for reproducibility
    static std::mt19937 gen(42);

    std::uniform_int_distribution<int> dist_type(0, 19);
    std::uniform_real_distribution<T> dist_param(0.5, 2.0);
    std::uniform_real_distribution<T> dist_pos(0.2, 0.8);

    T* ptr = data.data();

    auto get_wavelet_val = [&](int type, T t, T mu, T s, T w) -> T {
        T x = t - mu;
        T val = 0;
        T s2 = s * 2.0;

        switch(type) {
            case 0: val = std::cos(w*x) * std::exp(-x*x/(2*s*s)); break;
            case 1: { T x2 = (x*x)/(s*s); val = (1.0 - x2) * std::exp(-x2/2.0); } break;
            case 2: val = std::exp(-x*x/(2*s*s)); break;
            case 3: val = -x * std::exp(-x*x/(2*s*s)); break;
            case 4: if (x >= -s && x < 0) val = 1.0; else if (x >= 0 && x < s) val = -1.0; else val = 0.0; break;
            case 5: if (std::abs(x) < 1e-5) val = std::cos(w*x); else val = (std::sin(x/s)/(x/s)) * std::cos(w*x); break;
            case 6: val = std::cos(w*x + 0.01*x*x) * std::exp(-x*x/(2*s*s)); break;
            case 7: val = 1.0 / (1.0 + x*x/(s*s)); break;
            case 8: val = 1.0 / std::cosh(x/s); break;
            case 9: val = (std::abs(x) < s) ? 1.0 : 0.0; break;
            case 10: val = std::max((T)0.0, (T)1.0 - std::abs(x)/s); break;
            case 11: val = std::exp(-x*x/(2*s*s)) - 0.5 * std::exp(-x*x/(2*s2*s2)); break;
            case 12: if (std::abs(x) < 1e-5) val = 1.0; else { T sn = std::sin(x/s)/(x/s); val = sn*sn; } break;
            case 13: { T xt = x + s*2; if (xt > 0) val = xt * std::exp(-xt/s) * std::cos(w*xt); else val = 0; } break;
            case 14: val = std::exp(-x*x/(2*s*s)) * std::cos(5.0*x/s); break;
            case 15: { T xt = x + s; if (xt > 0) val = xt * std::exp(-xt/s); else val = 0; } break;
            case 16: { T xn = x/s; if (std::abs(xn) < 1.0) val = std::pow(1.0 - xn*xn, 2) * std::cos(w*x); else val = 0; } break;
            case 17: { T z = x/s; val = (8*z*z*z - 12*z) * std::exp(-z*z/2); } break;
            case 18: if (std::abs(x) < s) val = x/s; else val = 0; break;
            case 19: val = (std::cos(w*x) + 0.5*std::cos(2*w*x) + 0.25*std::cos(3*w*x)) * std::exp(-x*x/(2*s*s)); break;
        }
        return val;
    };

    for(size_t n=0; n<batch; ++n) {
        for(size_t c=0; c<C; ++c) {
            int type_x = dist_type(gen);
            int type_y = dist_type(gen);
            T s_x = dist_param(gen) * (W/10.0);
            T s_y = dist_param(gen) * (H/10.0);
            T mu_x = dist_pos(gen) * W;
            T mu_y = dist_pos(gen) * H;
            T w = dist_param(gen);

            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T wx = get_wavelet_val(type_x, (T)w_idx, mu_x, s_x, w);
                     T wy = get_wavelet_val(type_y, (T)h, mu_y, s_y, w);
                     ptr[((n*H + h)*W + w_idx)*C + c] = wx * wy;
                }
            }
        }
    }
}

void run_benchmark_for_channel(size_t C) {
    size_t batch_size = 8;
    size_t H = 64, W = 64;
    size_t latent_dim = 64;
    size_t loops = 10;

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Benchmarking Channel Count C=" << C << std::endl;

    // --- Zenith Model (Manual Chain) ---
    // Architecture: Pack -> Zenith -> Zenith -> Unpack -> Pool -> Pack -> Zenith -> Unpack ...
    // Simplified per user request architecture: Zenith -> Pool -> Zenith -> Latent -> ...
    // ZenithBlock is strictly int8 I/O. Pool/Dense are float.
    // Chain:
    // In(F) -> Pack -> Z(I8) -> Unpack -> Pool(F) -> Pack -> Z(I8) -> Unpack -> Flatten -> Dense -> Dense -> Reshape -> Pack -> Z(I8) -> Unpack -> Upscale -> Pack -> Z(I8) -> Unpack(Out)

    size_t arena_size = 4 * 1024 * 1024;

    // Layers
    layers::PackAPoT pack;
    layers::UnpackAPoT unpack;
    layers::ZenithBlock z1(C, 3, C, arena_size);
    AvgPool2D<float> pool(2); // In global namespace
    layers::ZenithBlock z2(C, 3, C, arena_size);
    Flatten<float> flat;
    layers::Dense<float> d1((H/2)*(W/2)*C, latent_dim);
    layers::Dense<float> d2(latent_dim, (H/2)*(W/2)*C);
    Reshape<float> reshape({H/2, W/2, C});
    layers::ZenithBlock z3(C, 3, C, arena_size);
    Upscale2D<float> upscale(2);
    layers::ZenithBlock z4(C, 3, C, arena_size);

    // --- Conv Model (Standard Sequential) ---
    layers::Conv2D<float> c1(C, C, 3, 1, 1);
    layers::Conv2D<float> c2(C, C, 3, 1, 1);
    layers::Conv2D<float> c3(C, C, 3, 1, 1);
    layers::Conv2D<float> c4(C, C, 3, 1, 1);

    // Data
    Tensor<float> input({batch_size, H, W, C});
    generate_wavelet_images(input);

    // Warmup Zenith Chain
    {
        auto t_packed = pack.forward(input);
        auto t_z1 = z1.forward(t_packed);
        // Pool requires float
        auto t_z1_f = unpack.forward(t_z1);
        auto t_pool = pool.forward(t_z1_f);
        auto t_pool_p = pack.forward(t_pool);
        auto t_z2 = z2.forward(t_pool_p);
        auto t_z2_f = unpack.forward(t_z2);
        auto t_flat = flat.forward(t_z2_f);
        auto t_d1 = d1.forward(t_flat);
        auto t_d2 = d2.forward(t_d1);
        auto t_resh = reshape.forward(t_d2);
        auto t_resh_p = pack.forward(t_resh);
        auto t_z3 = z3.forward(t_resh_p);
        auto t_z3_f = unpack.forward(t_z3);
        auto t_up = upscale.forward(t_z3_f);
        auto t_up_p = pack.forward(t_up);
        auto t_z4 = z4.forward(t_up_p);
        auto t_out = unpack.forward(t_z4);
    }

    // Benchmark Zenith
    auto start_z = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t_packed = pack.forward(input);
        auto t_z1 = z1.forward(t_packed);
        auto t_z1_f = unpack.forward(t_z1); // Explicit Unpack for Pooling
        auto t_pool = pool.forward(t_z1_f);
        auto t_pool_p = pack.forward(t_pool); // Pack again
        auto t_z2 = z2.forward(t_pool_p);
        auto t_z2_f = unpack.forward(t_z2); // Unpack for Dense
        auto t_flat = flat.forward(t_z2_f);
        auto t_d1 = d1.forward(t_flat);
        auto t_d2 = d2.forward(t_d1);
        auto t_resh = reshape.forward(t_d2);
        auto t_resh_p = pack.forward(t_resh); // Pack for Zenith
        auto t_z3 = z3.forward(t_resh_p);
        auto t_z3_f = unpack.forward(t_z3); // Unpack for Upscale
        auto t_up = upscale.forward(t_z3_f);
        auto t_up_p = pack.forward(t_up); // Pack for Zenith
        auto t_z4 = z4.forward(t_up_p);
        auto t_out = unpack.forward(t_z4); // Final Output
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double time_z = std::chrono::duration<double>(end_z - start_z).count();

    // Benchmark Conv
    auto start_c = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = c1.forward(input);
        auto t2 = pool.forward(t1);
        auto t3 = c2.forward(t2);
        auto t4 = flat.forward(t3);
        auto t5 = d1.forward(t4);
        auto t6 = d2.forward(t5);
        auto t7 = reshape.forward(t6);
        auto t8 = c3.forward(t7);
        auto t9 = upscale.forward(t8);
        auto t10 = c4.forward(t9);
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    double time_c = std::chrono::duration<double>(end_c - start_c).count();

    std::cout << "Zenith Time: " << time_z << " s" << std::endl;
    std::cout << "Conv2D Time: " << time_c << " s" << std::endl;
    std::cout << "Speedup: " << time_c / time_z << "x" << std::endl;
}

int main() {
    std::cout << "=== ZenithBlock (APoT) vs Conv2D Benchmark ===" << std::endl;
    std::vector<size_t> channels_list = {16, 64, 128};
    for(size_t C : channels_list) {
        run_benchmark_for_channel(C);
    }
    return 0;
}
