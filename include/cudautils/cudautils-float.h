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
    DMT_CPU_GPU inline constexpr float piOver4() { return 0.25f * std::numbers::pi_v<float>; }
    DMT_CPU_GPU inline constexpr float piOver2() { return 0.5f * std::numbers::pi_v<float>; }
    DMT_CPU_GPU inline constexpr float minNormalized() { return 1.175494351e-38f; }
    DMT_CPU_GPU inline constexpr float rcpPi() { return 1.f / std::numbers::pi_v<float>; }
    /** Light speed (vacuum) */
    DMT_CPU_GPU inline constexpr float c() { return 299792458.f; };
    /** Planck's constant */
    DMT_CPU_GPU inline constexpr float planck() { return 6.62606957e-34f; }
    /** Boltzmann's constant */
    DMT_CPU_GPU inline constexpr float kBoltz() { return 1.3806488e-23f; }

    DMT_CPU_GPU inline constexpr float degFromRad() { return 57.295779513082320876798154814105; }

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

    DMT_CPU_GPU inline float sqr(float v) { return v * v; }

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

    DMT_CPU_GPU inline float lerp(float delta, float a, float b)
    {
        if (delta == 0.f)
            return a;

#if defined(__CUDA_ARCH__)
        // On CUDA, just use fused multiply-add for better precision
        return __fmaf_rn(b - a, delta, a);
#else
    // On CPU: compute (b - a) * delta + a with FMA if available
    // But to guarantee exact return of `a` when delta == 0,
    // we do the if check above.
    #if defined(__FMA__)
        return std::fma(b - a, delta, a);
    #else
        return a + (b - a) * delta;
    #endif
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

    DMT_CPU_GPU inline float sqrt(float a)
    {
#if defined(__CUDA_ARCH__)
        return sqrtf(a);
#else
        return std::sqrt(a);
#endif
    }

    DMT_CPU_GPU inline float safeSqrt(float a)
    {
#if defined(__CUDA_ARCH__)
        return sqrtf(fmaxf(a, 0.f));
#else
        return std::sqrt(std::fmaxf(a, 0.f));
#endif
    }

    DMT_CPU_GPU inline float FMARoundUp(float a, float b, float c)
    {
#if defined(__CUDA_ARCH__)
        return __fma_ru(a, b, c);
#else
        return nextFloatUp(std::fmaf(a, b, c));
#endif
    }

    DMT_CPU_GPU inline float FMA(float a, float b, float c)
    {
#if defined(__CUDA_ARCH__)
        return fmaf(a, b, c);
#else
        return std::fmaf(a, b, c);
#endif
    }

    DMT_CPU_GPU inline float FMARoundDown(float a, float b, float c)
    {
#if defined(__CUDA_ARCH__)
        return __fma_rd(a, b, c);
#else
        return nextFloatDown(std::fmaf(a, b, c));
#endif
    }

    DMT_CPU_GPU inline float clamp(float x, float min, float max)
    {
#if defined(__CUDA_ARCH__)
        return ::min(::max(x, min), max);
#else
        return std::clamp(x, min, max);
#endif
    }

    DMT_CPU_GPU inline float clamp01(float x) { return clamp(x, 0, 1); }

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

    DMT_CPU_GPU inline float sign(float x)
    {
        if (x == 0.f || x == -0.f) // TODO tolerance?
            return 0.f;
#if defined(__CUDA_ARCH__)
        return ::signbit(x) ? -1.f : 1.f;
#else
        return std::signbit(x) ? -1.f : 1.f;
#endif
    }

    DMT_CPU_GPU inline float round(float x)
    {

#if defined(__CUDA_ARCH__)
        return ::roundf(x);
#else
        return std::roundf(x);
#endif
    }

    DMT_CORE_API DMT_CPU_GPU inline float xorf(float f, int32_t mask) { return bitsToFloat(floatToBits(f) & mask); }

    DMT_CORE_API DMT_CPU_GPU float rsqrt(float x);
    DMT_CORE_API DMT_CPU_GPU float abs(float x);
    DMT_CORE_API DMT_CPU_GPU float rcp(float x);
    DMT_CORE_API DMT_CPU_GPU bool  nearZero(float x, float tol = eqTol());
    DMT_CORE_API DMT_CPU_GPU bool  near(float x, float y);
    // Helper to compute (a^2 + b^2)^1/2 without overflow or underflow
    DMT_CORE_API DMT_CPU_GPU float pythag(float a, float b);

    DMT_CPU_GPU inline float safeacos(float v) { return clamp(acosf(v), -1, 1); }
} // namespace dmt::fl

namespace dmt {
    DMT_CPU_GPU DMT_CORE_API inline float smoothstep(float x)
    {
        if (x <= 0.f)
            return 0.f;
        if (x >= 1.f)
            return 1.f;
        float const x2 = x * x;
        return 3.f * x2 - 2.f * x2 * x;
    }

    DMT_CPU_GPU DMT_CORE_API inline float smoothstep(float a, float b, float x)
    {
        float const t = fl::clamp01((x - a) / (b - a));

        return smoothstep(t);
    }

    // TODO SOA
    struct DMT_CORE_API Intervalf
    {
        struct DMT_CORE_API SOA{};

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

    union FP32
    {
        uint32_t u;
        float    f;
        struct
        {
            unsigned int Mantissa : 23;
            unsigned int Exponent : 8;
            unsigned int Sign     : 1;
        };
    };

    union FP16
    {
        uint16_t u;
        struct
        {
            unsigned int Mantissa : 10;
            unsigned int Exponent : 5;
            unsigned int Sign     : 1;
        };
    };

    class Half
    {
    public:
        Half()                       = default;
        Half(Half const&)            = default;
        Half& operator=(Half const&) = default;

        DMT_CPU_GPU static Half FromBits(uint16_t v) { return Half(v); }

        explicit Half(float ff)
        {
#ifdef __CUDA_ARCH__
            h = __half_as_ushort(__float2half(ff));
#else
            // Rounding ties to nearest even instead of towards +inf
            FP32 f;
            f.f                       = ff;
            FP32         f32infty     = {255 << 23};
            FP32         f16max       = {(127 + 16) << 23};
            FP32         denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
            unsigned int sign_mask    = 0x80000000u;
            FP16         o            = {0};

            unsigned int sign = f.u & sign_mask;
            f.u ^= sign;

            // NOTE all the integer compares in this function can be safely
            // compiled into signed compares since all operands are below
            // 0x80000000. Important if you want fast straight SSE2 code
            // (since there's no unsigned PCMPGTD).

            if (f.u >= f16max.u)                            // result is Inf or NaN (all exponent bits set)
                o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
            else
            { // (De)normalized number or zero
                if (f.u < (113 << 23))
                { // resulting FP16 is subnormal or zero
                    // use a magic value to align our 10 mantissa bits at the bottom
                    // of the float. as long as FP addition is round-to-nearest-even
                    // this just works.
                    f.f += denorm_magic.f;

                    // and one integer subtract of the bias later, we have our final
                    // float!
                    o.u = f.u - denorm_magic.u;
                }
                else
                {
                    unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

                    // update exponent, rounding bias part 1
                    f.u += (uint32_t(15 - 127) << 23) + 0xfff;
                    // rounding bias part 2
                    f.u += mant_odd;
                    // take the bits!
                    o.u = f.u >> 13;
                }
            }

            o.u |= sign >> 16;
            h = o.u;
#endif
        }
        DMT_CPU_GPU
        explicit Half(double d) : Half(float(d)) {}

        DMT_CPU_GPU
        explicit operator float() const
        {
#ifdef __CUDA_ARCH__
            return __half2float(__ushort_as_half(h));
#else
            FP16 h;
            h.u                                   = this->h;
            static const FP32         magic       = {113 << 23};
            static unsigned int const shifted_exp = 0x7c00 << 13; // exponent mask after shift
            FP32                      o;

            o.u              = (h.u & 0x7fff) << 13; // exponent/mantissa bits
            unsigned int exp = shifted_exp & o.u;    // just the exponent
            o.u += (127 - 15) << 23;                 // exponent adjust

            // handle exponent special cases
            if (exp == shifted_exp)      // Inf/NaN?
                o.u += (128 - 16) << 23; // extra exp adjust
            else if (exp == 0)
            {                   // Zero/Denormal?
                o.u += 1 << 23; // extra exp adjust
                o.f -= magic.f; // renormalize
            }

            o.u |= (h.u & 0x8000) << 16; // sign bit
            return o.f;
#endif
        }
        DMT_CPU_GPU
        explicit operator double() const { return (float)(*this); }

        DMT_CPU_GPU
        bool operator==(Half const& v) const
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return __ushort_as_half(h) == __ushort_as_half(v.h);
#else
            if (Bits() == v.Bits())
                return true;
            return ((Bits() == HalfNegativeZero && v.Bits() == HalfPositiveZero) ||
                    (Bits() == HalfPositiveZero && v.Bits() == HalfNegativeZero));
#endif
        }
        DMT_CPU_GPU
        bool operator!=(Half const& v) const { return !(*this == v); }

        DMT_CPU_GPU
        Half operator-() const { return FromBits(h ^ (1 << 15)); }

        DMT_CPU_GPU
        uint16_t Bits() const { return h; }

        DMT_CPU_GPU
        int Sign() { return (h >> 15) ? -1 : 1; }

        DMT_CPU_GPU
        bool IsInf() { return h == HalfPositiveInfinity || h == HalfNegativeInfinity; }

        DMT_CPU_GPU
        bool IsNaN() { return ((h & HalfExponentMask) == HalfExponentMask && (h & HalfSignificandMask) != 0); }

        DMT_CPU_GPU
        Half NextUp()
        {
            if (IsInf() && Sign() == 1)
                return *this;

            Half up = *this;
            if (up.h == HalfNegativeZero)
                up.h = HalfPositiveZero;
            // Advance _v_ to next higher float
            if (up.Sign() >= 0)
                ++up.h;
            else
                --up.h;
            return up;
        }

        DMT_CPU_GPU
        Half NextDown()
        {
            if (IsInf() && Sign() == -1)
                return *this;

            Half down = *this;
            if (down.h == HalfPositiveZero)
                down.h = HalfNegativeZero;
            if (down.Sign() >= 0)
                --down.h;
            else
                ++down.h;
            return down;
        }

    private:
        static int const HalfExponentMask    = 0b0111110000000000;
        static int const HalfSignificandMask = 0b1111111111;
        static int const HalfNegativeZero    = 0b1000000000000000;
        static int const HalfPositiveZero    = 0;
        // Exponent all 1s, significand zero
        static int const HalfNegativeInfinity = 0b1111110000000000;
        static int const HalfPositiveInfinity = 0b0111110000000000;

        DMT_CPU_GPU
        explicit Half(uint16_t h) : h(h) {}

        uint16_t h;
    };

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
