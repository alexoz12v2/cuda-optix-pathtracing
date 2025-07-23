#pragma once

#include "cudautils/cudautils-macro.h"

#include <algorithm>
#include <bit>
#include <numbers>

#include <cmath>
#include <cstdint>
#include <limits>

// a present from Windows.h
#if defined(DMT_OS_WINDOWS)
    #pragma push_macro("near")
    #undef near
#endif

// TODO: If you want to rely to link time optimization, split definition and declaration and remove inline linkage
namespace dmt::fl {
    using namespace dmt;
    /** largest possible floating point less than 1 */
    DMT_CPU_GPU inline constexpr float oneMinusEps() { return 0x1.fffffep-1; }
    DMT_CPU_GPU inline constexpr float infinity() { return std::numeric_limits<float>::infinity(); }
    DMT_CPU_GPU inline constexpr float eqTol() { return std::numeric_limits<float>::epsilon(); }
    DMT_CPU_GPU inline constexpr float machineEpsilon() { return std::numeric_limits<float>::epsilon() * 0.5; }
    DMT_CPU_GPU inline constexpr float pi() { return std::numbers::pi_v<float>; }
    DMT_CPU_GPU inline constexpr float twoPi() { return 2.f * std::numbers::pi_v<float>; }
    /** Light speed (vacuum) */
    DMT_CPU_GPU inline constexpr float c() { return 299792458.f; };
    /** Planck's constant */
    DMT_CPU_GPU inline constexpr float planck() { return 6.62606957e-34f; }
    /** Boltzmann's constant */
    DMT_CPU_GPU inline constexpr float kBoltz() { return 1.3806488e-23f; }

    DMT_CPU_GPU inline bool isinf(float f)
    {
#if defined(__CUDA_ARCH__)
        return __isinff(f); // the last f stands for float32
#else
        return std::isinf(f);
#endif
    }

    DMT_CPU_GPU inline bool isNaN(float f)
    {
#if defined(__CUDA_ARCH__)
        return __isnanf(f);
#else
        return std::isnan(f);
#endif
    }

    DMT_CPU_GPU inline bool isInfOrNaN(float f)
    {
#if defined(__CUDA_ARCH__)
        return __isinff(f) || __isnanf(f);
#else
        return std::isinf(f) || std::isnan(f);
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

    DMT_CPU_GPU inline float asinClamp(float x)
    {
#if defined(__CUDA_ARCH__)
        return ::asin(::min(::max(x, -1.f), 1.f));
#else
        return std::asinf(std::clamp(x, -1.f, 1.f));
#endif
    }

    DMT_CPU_GPU inline float acosClamp(float x)
    {
#if defined(__CUDA_ARCH__)
        return ::asin(::min(::max(x, 0.f), 1.f));
#else
        return std::acosf(std::clamp(x, 0.f, 1.f));
#endif
    }

    DMT_CPU_GPU inline float copysign(float x, float y)
    {
#if defined(__CUDA_ARCH__)
        return ::copysign(x, y);
#else
        return std::copysign(x, y);
#endif
    }

    DMT_CPU_GPU inline float atan2(float y, float x)
    {
#if defined(__CUDA_ARCH__)
        return ::atan2(y, x);
#else
        return std::atan2f(y, x);
#endif
    }

    DMT_CORE_API DMT_CPU_GPU float rcp(float x);

    DMT_CORE_API DMT_CPU_GPU bool nearZero(float x);
    DMT_CORE_API DMT_CPU_GPU bool near(float x, float y);
    // Helper to compute (a^2 + b^2)^1/2 without overflow or underflow
    DMT_CORE_API DMT_CPU_GPU float pythag(float a, float b);

} // namespace dmt::fl

namespace dmt {
    // TODO SOA
    struct DMT_CORE_API Intervalf
    {
        struct DMT_CORE_API SOA
        {
        };

    public:
        Intervalf() = default;
        DMT_CPU_GPU explicit Intervalf(float v);
        DMT_CPU_GPU                  Intervalf(float low, float high);
        DMT_CPU_GPU static Intervalf fromValueAndError(float v, float err);

    public:
        DMT_CPU_GPU float midpoint() const;
        DMT_CPU_GPU float width() const;

    public:
        float low  = 0.f;
        float high = 0.f;
    };
    DMT_CPU_GPU Intervalf operator+(Intervalf a, Intervalf b);
    DMT_CPU_GPU Intervalf operator-(Intervalf a, Intervalf b);
    DMT_CPU_GPU Intervalf operator/(Intervalf a, Intervalf b);
    DMT_CPU_GPU Intervalf operator*(Intervalf a, Intervalf b);
} // namespace dmt

inline constexpr DMT_CPU_GPU float arg(float f)
{
    if (f > 0 || dmt::fl::floatToBits(f) == 0u)
        return 0;
    if (f < 0 || dmt::fl::floatToBits(f) == 0x8000'0000u)
        return dmt::fl::pi();
    else
        return std::numeric_limits<float>::quiet_NaN();
}

#if defined(DMT_OS_WINDOWS)
    #pragma pop_macro("near")
#endif

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_FLOAT_IMPL)
    #include "cudautils-float.cu"
#endif