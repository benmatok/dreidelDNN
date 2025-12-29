#ifndef DREIDEL_LAYERS_QUANTIZATION_HPP
#define DREIDEL_LAYERS_QUANTIZATION_HPP

#include "../core/Tensor.hpp"
#include "../hal/ops.hpp"
#include <vector>

namespace dreidel {
namespace layers {

// Explicit PACK layer: Float -> APoT Code
class PackAPoT {
public:
    Tensor<int8_t> forward(const Tensor<float>& input) {
        auto shape = input.shape();
        Tensor<int8_t> output(shape);

        size_t n = input.size();
        const float* in_ptr = input.data();
        int8_t* out_ptr = output.data();

        size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
        // Vectorized Pack (8 floats -> 8 bytes)
        for(; i + 8 <= n; i += 8) {
            hal::AlienOps::vec_pack_apot_avx2(in_ptr + i, out_ptr + i);
        }
#endif
        // Scalar Fallback
        for(; i < n; ++i) {
            out_ptr[i] = hal::AlienOps::pack_apot(in_ptr[i]);
        }
        return output;
    }

    std::string name() const { return "PackAPoT"; }
};

// Explicit UNPACK layer: APoT Code -> Float
class UnpackAPoT {
public:
    Tensor<float> forward(const Tensor<int8_t>& input) {
        auto shape = input.shape();
        Tensor<float> output(shape);

        size_t n = input.size();
        const int8_t* in_ptr = input.data();
        float* out_ptr = output.data();

        size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
        // Vectorized Unpack (8 bytes -> 8 floats)
        for(; i + 8 <= n; i += 8) {
            __m256 v = hal::AlienOps::vec_unpack_apot_avx2(in_ptr + i);
            _mm256_storeu_ps(out_ptr + i, v);
        }
#endif
        // Scalar Fallback
        for(; i < n; ++i) {
            out_ptr[i] = hal::AlienOps::unpack_apot(in_ptr[i]);
        }
        return output;
    }

    std::string name() const { return "UnpackAPoT"; }
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_QUANTIZATION_HPP
