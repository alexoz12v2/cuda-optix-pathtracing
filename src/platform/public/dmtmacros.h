#ifndef DMT_PLATFORM_PUBLIC_DMTMACROS_H
#define DMT_PLATFORM_PUBLIC_DMTMACROS_H

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
// TODO reg, interface dll

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

#if defined(DMT_NEEDS_MODULE) && !defined(__CUDACC__)
#  define DMT_MODULE_EXPORT export
#else
#  define DMT_MODULE_EXPORT
#endif

#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
#  if defined(DMT_OS_WINDOWS) || defined(DMT_COMPILER_MSVC)
#    define DMT_API_IMPORT __declspec(dllimport)
#    define DMT_API_EXPORT __declspec(dllexport)
#  else
#    define DMT_API_IMPORT __attribute__((__visibility__("default")))
#    define DMT_API_EXPORT __attribute__((__visibility__("default")))
#  endif
#else
#  define DMT_API_IMPORT
#  define DMT_API_EXPORT
#endif

#if defined(__CUDACC__)
#  define DMT_FORCEINLINE __forceinline__
#elif defined(DMT_COMPILER_MSVC)
#  define DMT_FORCEINLINE __forceinline
#elif defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
#  define DMT_FORCEINLINE __attribute__((always_inline))
#endif

#if defined(__CUDACC__) || defined(DMT_DEBUG)
// CUDA NVCC doesn't support alternate calling conventions; default to nothing.
#  define DMT_FASTCALL

#elif defined(DMT_COMPILER_MSVC)
#  if defined(DMT_ARCH_X86_64)
// __vectorcall uses XMM/YMM registers for floats/doubles, ideal for SSE/AVX
#    define DMT_FASTCALL __vectorcall
#  else
// On 32-bit MSVC, __fastcall is the closest
#    define DMT_FASTCALL __fastcall
#  endif

#elif defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
#  if defined(DMT_ARCH_X86_64)
// GCC/Clang on x86_64 default to System V ABI which already passes in
// registers, but we can still optionally use regcall if supported.
#    if defined(__GNUC__) && !defined(__clang__)
// regcall is GCC-specific extension on x86_64
#      define DMT_FASTCALL __attribute__((regcall))
#    else
// Clang doesn't support regcall; fallback to default
#      define DMT_FASTCALL
#    endif
#  else
// On 32-bit x86, use fastcall if available
#    define DMT_FASTCALL __attribute__((fastcall))
#  endif

#else
// Unknown compiler — fallback to nothing
#  define DMT_FASTCALL
#endif

// Visual Studio doesn't seem to pick up that .cu files (tagged with Item Type
// CUDA C++) should understand CUDA syntax
#if !defined(__CUDACC__)
#  define __host__
#  define __device__
#  define __global__
#  define __constant__
#  define __managed__
#  if 0  // breaks libc++
#    define __noinline__
#  endif
#endif

#endif  // DMT_PLATFORM_PUBLIC_DMTMACROS_H
