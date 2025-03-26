#pragma once

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
#define DMT_MANAGED __managed__
#else
#define DMT_CPU
#define DMT_CPU_GPU
#define DMT_MANAGED
#define DMT_GPU
#endif

#if defined(DMT_NEEDS_MODULE0) && !defined(__NVCC__)
#define DMT_MODULE_EXPORT export
#else
#define DMT_MODULE_EXPORT
#endif

#if defined(DMT_OS_WINDOWS) || defined(DMT_COMPILER_MSVC)
#define DMT_API_IMPORT __declspec(dllimport)
#define DMT_API_EXPORT __declspec(dllexport)
#else
#define DMT_API_IMPORT __attribute__((__visibility__("default")))
#define DMT_API_EXPORT __attribute__((__visibility__("default")))
#endif

#if defined(__NVCC__)
#define DMT_FORCEINLINE __forceinline__
#elif defined(DMT_COMPILER_MSVC)
#define DMT_FORCEINLINE __forceinline
#elif defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
#define DMT_FORCEINLINE __attribute__((always_inline))
#endif

// Visual Studio doesn't seem to pick up that .cu files (tagged with Item Type CUDA C++)
// should understand CUDA syntax
#if !defined(__NVCC__)
#define __host__
#define __device__
#define __global__
#define __constant__
#define __managed__
#define __noinline__
#endif
