#ifndef DREIDEL_MODELS_ZENITH_AUTOENCODER_C32_HPP
#define DREIDEL_MODELS_ZENITH_AUTOENCODER_C32_HPP

#include <vector>
#include "../layers/Layer.hpp"
#include "../layers/ZenithBlock.hpp"
#include "../layers/Conv2D.hpp"

namespace dreidel {
namespace models {

// Locked Zenith Autoencoder Architecture (C=32)
// This architecture targets ~9x speedup over equivalent Conv2D (in inference contexts)
// Training speedup observed: ~3x.
inline std::vector<layers::Layer<float>*> build_zenith_autoencoder_c32() {
    std::vector<layers::Layer<float>*> model;

    // Encoder: Stride=4, Upscale=1 (Downsample)
    // 64x64x1 -> 16x16x32
    model.push_back(new layers::ZenithBlock<float>(1, 32, 3, 1, true, true, false, 4, 1));
    // 16x16x32 -> 4x4x32
    model.push_back(new layers::ZenithBlock<float>(32, 32, 3, 32, true, true, false, 4, 1));
    // 4x4x32 -> 1x1x16
    model.push_back(new layers::ZenithBlock<float>(32, 16, 3, 32, true, true, false, 4, 1));

    // Decoder: Stride=1, Upscale=4 (Upsample)
    // 1x1x16 -> 4x4x32
    model.push_back(new layers::ZenithBlock<float>(16, 32, 3, 16, true, true, false, 1, 4));
    // 4x4x32 -> 16x16x32
    model.push_back(new layers::ZenithBlock<float>(32, 32, 3, 32, true, true, false, 1, 4));
    // 16x16x32 -> 64x64x1
    model.push_back(new layers::ZenithBlock<float>(32, 1, 3, 32, true, true, false, 1, 4));

    return model;
}

} // namespace models
} // namespace dreidel

#endif // DREIDEL_MODELS_ZENITH_AUTOENCODER_C32_HPP
