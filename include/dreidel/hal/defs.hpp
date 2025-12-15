#ifndef DREIDEL_HAL_DEFS_HPP
#define DREIDEL_HAL_DEFS_HPP

// Detect Architecture
#if defined(__AVX512F__)
    #define DREIDEL_ARCH_AVX512
#elif defined(__AVX2__)
    #define DREIDEL_ARCH_AVX2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define DREIDEL_ARCH_ARM_NEON
#else
    #define DREIDEL_ARCH_GENERIC
#endif

// Detect OpenGL Support (Manual Flag for now)
#ifdef DREIDEL_USE_OPENGL
    #define DREIDEL_ARCH_OPENGL
#endif

#endif // DREIDEL_HAL_DEFS_HPP
