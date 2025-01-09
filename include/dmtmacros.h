#pragma once

#if !defined(DMT_INTERFACE_AS_HEADER)
#define DMT_MODULE_EXPORT export namespace
#else
#define DMT_MODULE_EXPORT namespace
#endif

#if defined(__NVCC__) // Test
#define DMT_INTERFACE
#elif defined(DMT_COMPILER_MSVC)
#define DMT_INTERFACE __declspec(novtable)
#else
#define DMT_INTERFACE
#endif

#if defined(__NVCC__)
#define DMT_RESTRICT __restrict__
#elif defined(DMT_COMPILER_MSVC)
#define DMT_RESTRICT __restrict
#elif defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
#define DMT_RESTRICT __restrict__
#else
#define DMT_RESTRICT
#endif
// TODO reg, interface dll

#if defined(__NVCC__)
#define DMT_CPU     __host__
#define DMT_CPU_GPU __host__ __device__
#define DMT_GPU     __device__
#else
#define DMT_CPU
#define DMT_CPU_GPU
#define DMT_GPU
#endif