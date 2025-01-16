#pragma once

#include "dmtmacros.h"

#include <bit>

#include <cmath>
#include <cstdint>
#include <limits>

// TODO: If you want to rely to link time optimization, split definition and declaration and remove inline linkage
namespace dmt::fl {
    using namespace dmt;
    DMT_CPU_GPU inline constexpr float eqTol() { return std::numeric_limits<float>::epsilon(); }
    DMT_CPU_GPU inline constexpr float machineEpsilon() { return std::numeric_limits<float>::epsilon() * 0.5; }

    DMT_CPU_GPU inline bool isinf(float f)
    {
#if defined(__CUDA_ARCH__)
        return __isinff(f); // the last f stands for float32
#else
        return std::isinf(f);
#endif
    }

    DMT_CPU_GPU inline uint32_t floatToBits(float f)
    {
#if defined(__CUDA_ARCH__)
        return __float_as_uint(f);
#else
        return std::bit_cast<uint32_t>(f);
#endif
    }

    DMT_CPU_GPU inline float bitsToFloat(uint32_t ui)
    {
#if defined(__CUDA_ARCH__)
        return __uint_as_float(ui);
#else
        return std::bit_cast<float>(ui);
#endif
    }


    DMT_CPU_GPU inline constexpr float gamma(int32_t n)
    {
        float f = n * machineEpsilon();
        return f / (1 - f);
    }

    DMT_CPU_GPU inline float nextFloatUp(float v)
    {
        // Handle infinity and negative zero for _NextFloatUp()_
        if (isinf(v) && v > 0.f)
            return v;
        if (v == -0.f)
            v = 0.f;

        // Advance _v_ to next higher float
        uint32_t ui = floatToBits(v);
        if (v >= 0)
            ++ui;
        else
            --ui;
        return bitsToFloat(ui);
    }

    DMT_CPU_GPU inline float nextFloatDown(float v)
    {
        // Handle infinity and positive zero for _NextFloatDown()_
        if (isinf(v) && v < 0.)
            return v;
        if (v == 0.f)
            v = -0.f;

        uint32_t ui = floatToBits(v);
        if (v > 0)
            --ui;
        else
            ++ui;

        return bitsToFloat(ui);
    }

    DMT_CPU_GPU inline int exponent(float v) { return (floatToBits(v) >> 23) - 127; }

    DMT_CPU_GPU inline int significand(float v) { return floatToBits(v) & ((1 << 23) - 1); }

    DMT_CPU_GPU inline uint32_t signBit(float v) { return floatToBits(v) & 0x80000000; }


    DMT_CPU_GPU inline float addRoundUp(float a, float b)
    {
#if defined(__CUDA_ARCH__)
        return __fadd_ru(a, b);
#else // CPU
        return nextFloatUp(a + b);
#endif
    }

    DMT_CPU_GPU inline float addRoundDown(float a, float b)
    {
#if defined(__CUDA_ARCH__)
        return __fadd_rd(a, b);
#else
        return nextFloatDown(a + b);
#endif
    }

    DMT_CPU_GPU inline float subRoundUp(float a, float b) { return addRoundUp(a, -b); }

    DMT_CPU_GPU inline float subRoundDown(float a, float b) { return addRoundDown(a, -b); }

    DMT_CPU_GPU inline float mulRoundUp(float a, float b)
    {
#if defined(__CUDA_ARCH__)
        return __fmul_ru(a, b);
#else
        return nextFloatUp(a * b);
#endif
    }

    DMT_CPU_GPU inline float mulRoundDown(float a, float b)
    {
#if defined(__CUDA_ARCH__)
        return __fmul_rd(a, b);
#else
        return nextFloatDown(a * b);
#endif
    }

    DMT_CPU_GPU inline float divRoundUp(float a, float b)
    {
#if defined(__CUDA_ARCH__)
        return __fdiv_ru(a, b);
#else
        return nextFloatUp(a / b);
#endif
    }

    DMT_CPU_GPU inline float divRoundDown(float a, float b)
    {
#if defined(__CUDA_ARCH__)
        return __fdiv_rd(a, b);
#else
        return nextFloatDown(a / b);
#endif
    }

    DMT_CPU_GPU inline float sqrtRoundUp(float a)
    {
#if defined(__CUDA_ARCH__)
        return __fsqrt_ru(a);
#else
        return nextFloatUp(std::sqrt(a));
#endif
    }

    DMT_CPU_GPU inline float sqrtRoundDown(float a)
    {
#if defined(__CUDA_ARCH__)
        return __fsqrt_rd(a);
#else
        return std::max<float>(0, nextFloatDown(std::sqrt(a)));
#endif
    }

    DMT_CPU_GPU inline float FMARoundUp(float a, float b, float c)
    {
#if defined(__CUDA_ARCH__)
        return __fma_ru(a, b, c);
#else
        return nextFloatUp(std::fma(a, b, c));
#endif
    }

    DMT_CPU_GPU inline float FMARoundDown(float a, float b, float c)
    {
#if defined(__CUDA_ARCH__)
        return __fma_rd(a, b, c);
#else
        return nextFloatDown(std::fma(a, b, c));
#endif
    }
} // namespace dmt::fl