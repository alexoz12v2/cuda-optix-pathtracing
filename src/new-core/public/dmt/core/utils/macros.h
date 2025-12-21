#pragma once

#include <cassert>

#if defined(__CUDACC__)
#  define DMT_INTERFACE
#elif defined(DMT_COMPILER_MSVC)
#  define DMT_INTERFACE __declspec(novtable)
#else
#  define DMT_INTERFACE
#endif

#if defined(__CUDACC__)
#  define DMT_RESTRICT __restrict__
#elif defined(DMT_COMPILER_MSVC)
#  define DMT_RESTRICT __restrict
#elif defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
#  define DMT_RESTRICT __restrict__
#else
#  define DMT_RESTRICT
#endif

// To be preferred when writing shared headers!
#if defined(__CUDACC__)
#  define DMT_CPU __host__
#  define DMT_CPU_GPU __host__ __device__
#  define DMT_GPU __device__
#  define DMT_MANAGED __managed__
#else
#  define DMT_CPU
#  define DMT_CPU_GPU
#  define DMT_MANAGED
#  define DMT_GPU
#endif

#ifdef DMT_DEBUG
#  ifdef __CUDA_ARCH__
#    define DMT_ENSURE(...)  // TODO trap? on which thread in the warp?
#  else
#    define DMT_ENSURE(...) assert(__VA_ARGS__)
#  endif
#else
#  define DMT_ENSURE(...)
#endif