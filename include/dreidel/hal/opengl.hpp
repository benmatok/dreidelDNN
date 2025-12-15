#ifndef DREIDEL_HAL_OPENGL_HPP
#define DREIDEL_HAL_OPENGL_HPP

#include <string>

namespace dreidel {
namespace hal {
namespace opengl {

// OpenGL Compute Shader for one step of FWHT
// Must be dispatched log2(N) times.
// 'step' is the current stage index (0 to log2(N)-1).
// pair_dist = 1 << step.
static const char* WHT_COMPUTE_SHADER = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    float values[];
};

uniform int N;          // Total size of the vector
uniform int pair_dist;  // 1 << step

void main() {
    uint idx = gl_GlobalInvocationID.x;
    // We process N/2 butterflies
    if (idx >= N / 2) return;

    // Map linear index 'idx' to butterfly indices (i, j)
    // The pattern depends on pair_dist.
    // Each block of 'pair_dist * 2' contains 'pair_dist' butterflies.
    // Within a block, indices are [0..pair_dist-1] (conceptually).

    int block_width = pair_dist * 2;
    int block_idx = int(idx) / pair_dist;
    int offset_in_block = int(idx) % pair_dist;

    int i = block_idx * block_width + offset_in_block;
    int j = i + pair_dist;

    float u = values[i];
    float v = values[j];

    values[i] = u + v;
    values[j] = u - v;
}
)";

struct Ops {
    // Dispatch logic stub
    template <typename T>
    static void fwht_dispatch(T* data, size_t N) {
        // This would interact with OpenGL API to:
        // 1. Create SSBO from data (or assume data is mapped).
        // 2. Use the shader program.
        // 3. Loop log2(N) times, setting uniforms and dispatching.
        //    int steps = log2(N);
        //    for (int s = 0; s < steps; ++s) {
        //        glUniform1i(loc_pair_dist, 1 << s);
        //        glDispatchCompute((N/2 + 255) / 256, 1, 1);
        //        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        //    }

        // Since we are in a header-only library without GL dependencies linked,
        // we throw an error or log a warning if this path is taken.
        throw std::runtime_error("OpenGL backend dispatch requires an active GL context and implementation linkage.");
    }
};

} // namespace opengl
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_OPENGL_HPP
