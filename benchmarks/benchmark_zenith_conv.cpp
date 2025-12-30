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

using namespace dreidel;

// Helpers: We only need standard float layers for the Conv model now.

template <typename T>
class AvgPool2D : public layers::Layer<T> {
public:
    AvgPool2D(size_t stride) : stride_(stride) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        // Assume NHWC
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
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
    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); }
    std::string name() const override { return "AvgPool2D"; }
private:
    size_t stride_;
};

template <typename T>
class Upscale2D : public layers::Layer<T> {
public:
    Upscale2D(size_t scale) : scale_(scale) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
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
};

// 2D Wavelet Generator (Rich Families)
template <typename T>
void generate_wavelet_images(Tensor<T>& data) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    static std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist_type(0, 19);
    std::uniform_real_distribution<T> dist_param(0.5, 2.0);
    std::uniform_real_distribution<T> dist_pos(0.2, 0.8);
    T* ptr = data.data();
    auto get_wavelet_val = [&](int type, T t, T mu, T s, T w) -> T {
        T x = t - mu; T val = 0; T s2 = s * 2.0;
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
            int type_x = dist_type(gen); int type_y = dist_type(gen);
            T s_x = dist_param(gen) * (W/10.0); T s_y = dist_param(gen) * (H/10.0);
            T mu_x = dist_pos(gen) * W; T mu_y = dist_pos(gen) * H;
            T w_p = dist_param(gen);
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T wx = get_wavelet_val(type_x, (T)w_idx, mu_x, s_x, w_p);
                     T wy = get_wavelet_val(type_y, (T)h, mu_y, s_y, w_p);
                     ptr[((n*H + h)*W + w_idx)*C + c] = wx * wy;
                }
            }
        }
    }
}

void run_benchmark_for_channel(size_t C) {
    size_t batch_size = 8;
    size_t H = 64, W = 64;
    size_t loops = 10;

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Benchmarking Channel Count C=" << C << std::endl;

    // Data
    Tensor<float> input({batch_size, H, W, C});
    generate_wavelet_images(input);

    // --- Zenith Model Configurations ---
    auto run_zenith = [&](const std::string& name, bool ifwht, bool dilated) {
        layers::ZenithBlock<float> z1(C, 3, C, ifwht, dilated);
        AvgPool2D<float> pool(2);
        layers::ZenithBlock<float> z2(C, 3, C, ifwht, dilated);
        layers::ZenithBlock<float> z3(C, 3, C, ifwht, dilated);
        Upscale2D<float> upscale(2);
        layers::ZenithBlock<float> z4(C, 3, C, ifwht, dilated);

        // Warmup
        {
            auto t1 = z1.forward(input);
            auto t2 = pool.forward(t1);
            auto t3 = z2.forward(t2);
            auto t4 = z3.forward(t3);
            auto t5 = upscale.forward(t4);
            auto t6 = z4.forward(t5);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<loops; ++i) {
            auto t1 = z1.forward(input);
            auto t2 = pool.forward(t1);
            auto t3 = z2.forward(t2);
            auto t4 = z3.forward(t3);
            auto t5 = upscale.forward(t4);
            auto t6 = z4.forward(t5);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        std::cout << name << " Time: " << time << " s" << std::endl;
        return time;
    };

    double t_base = run_zenith("Zenith (Base)", false, false);
    double t_ifwht = run_zenith("Zenith (+IFWHT)", true, false);
    double t_full = run_zenith("Zenith (+IFWHT+Dilated)", true, true);

    // --- Conv Model ---
    layers::Conv2D<float> c1(C, C, 3, 1, 1);
    layers::Conv2D<float> c2(C, C, 3, 1, 1);
    layers::Conv2D<float> c3(C, C, 3, 1, 1);
    layers::Conv2D<float> c4(C, C, 3, 1, 1);

    AvgPool2D<float> pool_f(2);
    Upscale2D<float> upscale_f(2);

    auto start_c = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = c1.forward(input);
        auto t2 = pool_f.forward(t1);
        auto t3 = c2.forward(t2);
        auto t4 = c3.forward(t3);
        auto t5 = upscale_f.forward(t4);
        auto t6 = c4.forward(t5);
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    double time_c = std::chrono::duration<double>(end_c - start_c).count();

    std::cout << "Conv2D Time: " << time_c << " s" << std::endl;
    std::cout << "Speedup (Full vs Conv): " << time_c / t_full << "x" << std::endl;
}

int main() {
    std::cout << "=== ZenithBlock (Float) vs Conv2D Benchmark ===" << std::endl;
    std::vector<size_t> channels_list = {16, 64, 128};
    for(size_t C : channels_list) {
        run_benchmark_for_channel(C);
    }
    return 0;
}
