#ifndef DMT_CUDA_CORE_DEBUG_CUH
#define DMT_CUDA_CORE_DEBUG_CUH

#include <cuda_runtime.h>
#include <cstdio>

// kernel debug print
#ifdef __CUDA_ARCH__
// #  define ISECT_PRINT(...) printf(__VA_ARGS__)
#  define ISECT_PRINT(...)
// #  define PRINT(...)                                                     \
//     do {                                                                 \
//       if (pixel.x == 895 / 2 && pixel.y == 246 / 2) printf(__VA_ARGS__); \
//     } while (0)
#  define PRINT(...)
#else
#  define PRINT(...) printf(__VA_ARGS__)
#  define ISECT_PRINT(...) printf(__VA_ARGS__)
#endif

#ifdef DMT_DEBUG
#  define DMT_ENABLE_ASSERTS 1
#else
#  define DMT_ENABLE_ASSERTS 0
#endif

#endif