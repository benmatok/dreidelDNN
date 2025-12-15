#ifndef DREIDEL_HAL_OPS_HPP
#define DREIDEL_HAL_OPS_HPP

#include "defs.hpp"
#include "generic.hpp"
#include "x86.hpp"
#include "arm.hpp"
#include "opengl.hpp"

namespace dreidel {
namespace hal {

// Select the active implementation based on macros
#if defined(DREIDEL_ARCH_AVX512) || defined(DREIDEL_ARCH_AVX2)
    using ActiveOps = x86::Ops;
#elif defined(DREIDEL_ARCH_ARM_NEON)
    using ActiveOps = arm::Ops;
#else
    using ActiveOps = generic::Ops;
#endif

} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_OPS_HPP
