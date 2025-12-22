#ifndef DREIDEL_ALGO_FFT_HPP
#define DREIDEL_ALGO_FFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dreidel {
namespace algo {

class FFT {
public:
    using Complex = std::complex<float>;

    // Simple Cooley-Tukey FFT (Iterative)
    // Input size must be power of 2
    static void fft(std::vector<Complex>& a) {
        size_t n = a.size();
        if (n <= 1) return;

        // Bit-reversal permutation
        size_t j = 0;
        for (size_t i = 1; i < n; ++i) {
            size_t bit = n >> 1;
            while (j & bit) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j) {
                std::swap(a[i], a[j]);
            }
        }

        // Butterfly updates
        for (size_t len = 2; len <= n; len <<= 1) {
            float ang = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
            Complex wlen(std::cos(ang), std::sin(ang));
            for (size_t i = 0; i < n; i += len) {
                Complex w(1);
                for (size_t k = 0; k < len / 2; ++k) {
                    Complex u = a[i + k];
                    Complex v = a[i + k + len / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    // Real-to-Complex FFT
    // Takes real input, pads to next power of 2 if needed (or assumes caller did),
    // and returns the positive frequency half.
    static void rfft(const std::vector<float>& input, std::vector<Complex>& output) {
         size_t n = input.size();
         // Ensure power of 2
         size_t p2 = 1;
         while (p2 < n) p2 <<= 1;

         output.assign(p2, Complex(0, 0));
         for(size_t i=0; i<n; ++i) output[i] = Complex(input[i], 0);

         fft(output);

         // Resize to n/2 + 1
         output.resize(p2/2 + 1);
    }
};

} // namespace algo
} // namespace dreidel

#endif // DREIDEL_ALGO_FFT_HPP
