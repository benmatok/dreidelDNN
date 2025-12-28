#ifndef DREIDEL_UTILS_WAVELETGEN_HPP
#define DREIDEL_UTILS_WAVELETGEN_HPP

#include <vector>
#include <random>
#include <cmath>
#include "../core/Tensor.hpp"

namespace dreidel {
namespace utils {

// Mixed Wavelet Generator (20 Types)
// Uses owning Tensor for offline data generation
template <typename T>
void generate_mixed_wavelets(dreidel::Tensor<T>& data, size_t batch_size, size_t dim) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Parameters
    std::uniform_real_distribution<T> dist_A(0.5, 1.5);
    std::uniform_real_distribution<T> dist_mu(dim * 0.2, dim * 0.8);
    std::uniform_real_distribution<T> dist_s(2.0, 15.0);
    std::uniform_real_distribution<T> dist_w(0.1, 0.8);
    std::uniform_real_distribution<T> dist_phi(0.0, 6.28);
    std::uniform_int_distribution<int> dist_type(0, 19);

    T* ptr = data.data();

    for (size_t i = 0; i < batch_size; ++i) {
        int type = dist_type(gen);
        T A = dist_A(gen);
        T mu = dist_mu(gen);
        T s = dist_s(gen);
        T w = dist_w(gen);
        T phi = dist_phi(gen);

        // Secondary scale for DoG etc
        T s2 = s * 2.0;

        for (size_t t = 0; t < dim; ++t) {
            T x = static_cast<T>(t) - mu;
            T val = 0;

            switch(type) {
                case 0: // Gabor
                    val = std::cos(w*x + phi) * std::exp(-x*x/(2*s*s));
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
                case 4: // Haar-like
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
                case 11: // DoG
                    val = std::exp(-x*x/(2*s*s)) - 0.5 * std::exp(-x*x/(2*s2*s2));
                    break;
                case 12: // Sinc Squared
                    if (std::abs(x) < 1e-5) val = 1.0;
                    else {
                        T sn = std::sin(x/s)/(x/s);
                        val = sn*sn;
                    }
                    break;
                case 13: // Gammatone
                    {
                        T xt = x + s*2;
                        if (xt > 0) val = xt * std::exp(-xt/s) * std::cos(w*xt);
                        else val = 0;
                    }
                    break;
                case 14: // Morlet (Real)
                    val = std::exp(-x*x/(2*s*s)) * std::cos(5.0*x/s);
                    break;
                case 15: // Poisson Wavelet
                    {
                        T xt = x + s;
                        if (xt > 0) val = xt * std::exp(-xt/s);
                        else val = 0;
                    }
                    break;
                case 16: // Beta Wavelet
                    {
                        T xn = x/s;
                        if (std::abs(xn) < 1.0) {
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
                case 19: // Random Walk
                    val = std::cos(w*x) + 0.5*std::cos(2*w*x) + 0.25*std::cos(3*w*x);
                    val *= std::exp(-x*x/(2*s*s));
                    break;
            }

            ptr[i * dim + t] = A * val;
        }
    }
}

} // namespace utils
} // namespace dreidel

#endif // DREIDEL_UTILS_WAVELETGEN_HPP
