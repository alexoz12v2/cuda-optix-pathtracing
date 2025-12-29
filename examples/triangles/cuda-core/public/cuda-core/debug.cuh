#ifndef DMT_CUDA_CORE_DEBUG_CUH
#define DMT_CUDA_CORE_DEBUG_CUH

#include <cuda_runtime.h>
#include <cstdio>

// kernel debug print
#ifdef __CUDA_ARCH__
// #  define ISECT_PRINT(...) printf(__VA_ARGS__)
#  define ISECT_PRINT(...)
// #  define PRINT(...) printf(__VA_ARGS__)
#  define PRINT(...)
#else
#  define PRINT(...) printf(__VA_ARGS__)
#  define ISECT_PRINT(...) printf(__VA_ARGS__)
#endif

#endif