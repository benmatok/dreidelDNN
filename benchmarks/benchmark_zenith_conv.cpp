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
#include "../include/dreidel/layers/GELU.hpp"

using namespace dreidel;

// Helpers
template <typename T>
class AvgPool2D : public layers::Layer<T> {
public:
    AvgPool2D(size_t stride) : stride_(stride) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        // Assume NHWC
        input_shape_ = input.shape();
        size_t N = input_shape_[0];
        size_t H = input_shape_[1];
        size_t W = input_shape_[2];
        size_t C = input_shape_[3];

        size_t H_out = H / stride_;
        size_t W_out = W / stride_;

        Tensor<T> output({N, H_out, W_out, C});
        output.fill(0);
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

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

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        size_t N = input_shape_[0];
        size_t H = input_shape_[1];
        size_t W = input_shape_[2];
        size_t C = input_shape_[3];
        size_t H_out = H / stride_;
        size_t W_out = W / stride_;

        Tensor<T> grad_input(input_shape_);
        grad_input.fill(0);
        T* gi_ptr = grad_input.data();
        const T* go_ptr = grad_output.data();

        T scale = 1.0 / (stride_ * stride_);

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    for(size_t c=0; c<C; ++c) {
                        T dy = go_ptr[((n*H_out + h)*W_out + w)*C + c];
                        for(size_t dy_off=0; dy_off<stride_; ++dy_off) {
                            for(size_t dx_off=0; dx_off<stride_; ++dx_off) {
                                gi_ptr[((n*H + h*stride_+dy_off)*W + w*stride_+dx_off)*C + c] = dy * scale;
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
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
        size_t N = input_shape_[0];
        size_t H = input_shape_[1];
        size_t W = input_shape_[2];
        size_t C = input_shape_[3];

        size_t H_out = H * scale_;
        size_t W_out = W * scale_;

        Tensor<T> output({N, H_out, W_out, C});
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_;
                    size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        out_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = in_ptr[((n*H + h_in)*W + w_in)*C + c];
                    }
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        size_t N = input_shape_[0];
        size_t H = input_shape_[1];
        size_t W = input_shape_[2];
        size_t C = input_shape_[3];
        size_t H_out = H * scale_;
        size_t W_out = W * scale_;

        Tensor<T> grad_input(input_shape_);
        grad_input.fill(0);
        T* gi_ptr = grad_input.data();
        const T* go_ptr = grad_output.data();

        // Accumulate gradients from upscaled pixels back to source
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_;
                    size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        gi_ptr[((n*H + h_in)*W + w_in)*C + c] += go_ptr[((n*H_out + h_out)*W_out + w_out)*C + c];
                    }
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
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

        // Tensor copy but reshaped
        Tensor<T> output({batch, dim});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();
        std::copy(in_ptr, in_ptr + output.size(), out_ptr);
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad_input(input_shape_);
        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();
        std::copy(go_ptr, go_ptr + grad_input.size(), gi_ptr);
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
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
        const T* in_ptr = input.data();
        T* out_ptr = output.data();
        std::copy(in_ptr, in_ptr + output.size(), out_ptr);
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad_input(input_shape_);
        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();
        std::copy(go_ptr, go_ptr + grad_input.size(), gi_ptr);
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::string name() const override { return "Reshape"; }
private:
    std::vector<size_t> input_shape_;
    std::vector<size_t> target_suffix_;
};

// Simple Sequential Container
template <typename T>
class Sequential {
public:
    ~Sequential() { for(auto l : layers_) delete l; }
    void add(layers::Layer<T>* l) { layers_.push_back(l); }

    Tensor<T> forward(const Tensor<T>& x) {
        Tensor<T> out = x;
        for(auto l : layers_) out = l->forward(out);
        return out;
    }

    // Minimal params exposure
    size_t param_count() {
        size_t cnt = 0;
        for(auto l : layers_) {
            for(auto p : l->parameters()) cnt += p->size();
        }
        return cnt;
    }
private:
    std::vector<layers::Layer<T>*> layers_;
};

// 2D Wavelet Generator
// Adapts 1D mixed wavelets to 2D by product or radial
template <typename T>
void generate_wavelet_images(Tensor<T>& data) {
    auto shape = data.shape();
    size_t batch = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    // Use fixed seed for reproducibility
    static std::mt19937 gen(42);

    // We will generate 2D patterns.
    // Simple approach: Product of two 1D wavelets for X and Y, modulated across channels

    std::uniform_int_distribution<int> dist_type(0, 19);
    std::uniform_real_distribution<T> dist_param(0.5, 2.0);
    std::uniform_real_distribution<T> dist_pos(0.2, 0.8);

    T* ptr = data.data();

    // Wavelet 1D function (Full Rich Families from Autoencoder benchmark)
    auto get_wavelet_val = [&](int type, T t, T mu, T s, T w) -> T {
        T x = t - mu;
        T val = 0;
        T s2 = s * 2.0;

        switch(type) {
            case 0: // Gabor
                val = std::cos(w*x) * std::exp(-x*x/(2*s*s));
                break;
            case 1: // Mexican Hat (Ricker)
                {
                    T x2 = (x*x)/(s*s);
                    val = (1.0 - x2) * std::exp(-x2/2.0);
                }
                break;
            case 2: // Gaussian
                val = std::exp(-x*x/(2*s*s));
                break;
            case 3: // Gaussian Derivative 1
                val = -x * std::exp(-x*x/(2*s*s));
                break;
            case 4: // Haar-like (Smooth approximation or hard)
                // Hard Haar
                if (x >= -s && x < 0) val = 1.0;
                else if (x >= 0 && x < s) val = -1.0;
                else val = 0.0;
                break;
            case 5: // Shannon (Sinc * Cos)
                if (std::abs(x) < 1e-5) val = std::cos(w*x);
                else val = (std::sin(x/s)/(x/s)) * std::cos(w*x);
                break;
            case 6: // Chirp (Linear FM)
                val = std::cos(w*x + 0.01*x*x) * std::exp(-x*x/(2*s*s));
                break;
            case 7: // Lorentzian
                val = 1.0 / (1.0 + x*x/(s*s));
                break;
            case 8: // Sech
                val = 1.0 / std::cosh(x/s);
                break;
            case 9: // Boxcar (Rect)
                val = (std::abs(x) < s) ? 1.0 : 0.0;
                break;
            case 10: // Triangular
                val = std::max((T)0.0, (T)1.0 - std::abs(x)/s);
                break;
            case 11: // DoG (Difference of Gaussians)
                val = std::exp(-x*x/(2*s*s)) - 0.5 * std::exp(-x*x/(2*s2*s2));
                break;
            case 12: // Sinc Squared
                if (std::abs(x) < 1e-5) val = 1.0;
                else {
                    T sn = std::sin(x/s)/(x/s);
                    val = sn*sn;
                }
                break;
            case 13: // Gammatone (approx, x>0 part centered)
                {
                    T xt = x + s*2; // Shift to make visible
                    if (xt > 0) val = xt * std::exp(-xt/s) * std::cos(w*xt);
                    else val = 0;
                }
                break;
            case 14: // Morlet (Real)
                val = std::exp(-x*x/(2*s*s)) * std::cos(5.0*x/s);
                break;
            case 15: // Poisson Wavelet
                {
                    T xt = x + s; // shift
                    if (xt > 0) val = xt * std::exp(-xt/s);
                    else val = 0;
                }
                break;
            case 16: // Beta Wavelet
                {
                    T xn = x/s;
                    if (std::abs(xn) < 1.0) {
                        // (1-x^2) * cos...
                        val = std::pow(1.0 - xn*xn, 2) * std::cos(w*x);
                    } else val = 0;
                }
                break;
            case 17: // Hermite H3
                {
                    T z = x/s;
                    val = (8*z*z*z - 12*z) * std::exp(-z*z/2);
                }
                break;
            case 18: // Sawtooth Pulse
                if (std::abs(x) < s) val = x/s;
                else val = 0;
                break;
            case 19: // Random Walk (Brownian Bridge approx)
                // Hard to generate pointwise statelessly.
                // Use a functional approximation: Sum of 3 Cosines
                val = std::cos(w*x) + 0.5*std::cos(2*w*x) + 0.25*std::cos(3*w*x);
                val *= std::exp(-x*x/(2*s*s));
                break;
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

int main() {
    // Config
    size_t batch_size = 8;
    size_t H = 64, W = 64, C = 16;
    size_t latent_dim = 64;
    size_t loops = 10;

    std::cout << "=== ZenithBlock vs Conv2D Benchmark ===" << std::endl;
    std::cout << "Input: (" << batch_size << ", " << H << ", " << W << ", " << C << ")" << std::endl;
    std::cout << "Architecture: Block -> Pooling -> Block -> Latent(64) -> Block -> Upscale -> Block" << std::endl;

    // --- Zenith Model ---
    Sequential<float> zenith_net;
    // Zenith -> Pooling -> Zenith
    zenith_net.add(new layers::ZenithBlock<float>(C, 3, C));
    zenith_net.add(new AvgPool2D<float>(2)); // 32x32
    zenith_net.add(new layers::ZenithBlock<float>(C, 3, C));

    // -> Latent 64
    zenith_net.add(new Flatten<float>());
    size_t flat_dim = (H/2)*(W/2)*C;
    zenith_net.add(new layers::Dense<float>(flat_dim, latent_dim));

    // -> Transpose Zenith (Decoder)
    zenith_net.add(new layers::Dense<float>(latent_dim, flat_dim));
    zenith_net.add(new Reshape<float>({H/2, W/2, C}));
    zenith_net.add(new layers::ZenithBlock<float>(C, 3, C));
    zenith_net.add(new Upscale2D<float>(2)); // 64x64
    zenith_net.add(new layers::ZenithBlock<float>(C, 3, C));

    // --- Conv Model ---
    Sequential<float> conv_net;
    conv_net.add(new layers::Conv2D<float>(C, C, 3, 1, 1)); // Padding 1 to keep size
    conv_net.add(new AvgPool2D<float>(2));
    conv_net.add(new layers::Conv2D<float>(C, C, 3, 1, 1));

    conv_net.add(new Flatten<float>());
    conv_net.add(new layers::Dense<float>(flat_dim, latent_dim));

    conv_net.add(new layers::Dense<float>(latent_dim, flat_dim));
    conv_net.add(new Reshape<float>({H/2, W/2, C}));
    conv_net.add(new layers::Conv2D<float>(C, C, 3, 1, 1));
    conv_net.add(new Upscale2D<float>(2));
    conv_net.add(new layers::Conv2D<float>(C, C, 3, 1, 1));

    std::cout << "Zenith Params: " << zenith_net.param_count() << std::endl;
    std::cout << "Conv2D Params: " << conv_net.param_count() << std::endl;

    // Data
    Tensor<float> input({batch_size, H, W, C});
    generate_wavelet_images(input);

    // Warmup
    zenith_net.forward(input);
    conv_net.forward(input);

    // Benchmark Zenith
    auto start_z = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        Tensor<float> out = zenith_net.forward(input);
        // Force sync? CPU is sync.
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double time_z = std::chrono::duration<double>(end_z - start_z).count();

    // Benchmark Conv
    auto start_c = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        Tensor<float> out = conv_net.forward(input);
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    double time_c = std::chrono::duration<double>(end_c - start_c).count();

    std::cout << "\nResults (" << loops << " loops):" << std::endl;
    std::cout << "Zenith Time: " << time_z << " s (" << (loops/time_z) << " iter/s)" << std::endl;
    std::cout << "Conv2D Time: " << time_c << " s (" << (loops/time_c) << " iter/s)" << std::endl;

    double speedup = time_c / time_z;
    std::cout << "Speedup (Zenith vs Conv2D): " << speedup << "x" << std::endl;

    if (speedup > 1.0) {
        std::cout << "SUCCESS: ZenithBlock is faster!" << std::endl;
    } else {
        std::cout << "NOTE: ZenithBlock is slower. This might be due to OMP in Conv2D vs Serial Zenith." << std::endl;
    }

    return 0;
}
