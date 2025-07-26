#include "cudautils-float.h"

// TODO add platform-math
#if defined(DMT_ARCH_X86_64)
    #include <immintrin.h>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtc/epsilon.hpp>

#if defined(near)
    #undef near
#endif

namespace dmt::fl {
    __host__ __device__ float rcp(float x)
    {
        float ret = x;
#if !defined(DMT_SKIP_FLOAT_TESTS)
        if (glm::epsilonEqual(x, 0.f, eqTol()))
            return std::numeric_limits<float>::infinity();
#endif

#if defined(__CUDA_ARCH__)
        float approx;
        asm("mov.f32 %0, %1;" : "=f"(approx) : "f"(x));             // Move x into a register
        asm("rcp.approx.f32 %0, %0;" : "=f"(approx) : "f"(approx)); // Approximate reciprocal
        ret = approx;
#elif defined(DMT_ARCH_X86_64)
        __m128 data = _mm_set_ss(ret);
        data        = _mm_rcp_ss(data);
        _mm_store_ss(&ret, data);
#else
        ret = 1.f / x;
#endif
        return ret;
    }
    __host__ __device__ bool  nearZero(float x) { return glm::epsilonEqual(x, 0.f, eqTol()); }
    __host__ __device__ bool  near(float x, float y) { return glm::epsilonEqual(x, y, eqTol()); }
    __host__ __device__ float pythag(float a, float b)
    {
        float absa = glm::abs(a);
        float absb = glm::abs(b);
        if (absa > absb)
            return absa * glm::sqrt(1.0f + (absb / absa) * (absb / absa));
        else
            return (absb == 0.0f ? 0.0f : absb * glm::sqrt(1.0f + (absa / absb) * (absa / absb)));
    }
} // namespace dmt::fl

namespace dmt {
    // math utilities: float ------------------------------------------------------------------------------------------
    __host__ __device__ Intervalf::Intervalf(float v) : low(v), high(v) {}

    __host__ __device__ Intervalf::Intervalf(float low, float high) :
    low(glm::min(low, high)),
    high(glm::max(low, high))
    {
    }

    __host__ __device__ Intervalf Intervalf::fromValueAndError(float v, float err)
    {
        Intervalf i;
        if (err == 0.f)
            i.low = i.high = v;
        else
        {
            i.low  = fl::subRoundDown(v, err);
            i.high = fl::addRoundUp(v, err);
        }
        return i;
    }

    __host__ __device__ float Intervalf::midpoint() const { return (low + high) / 2; }

    __host__ __device__ float Intervalf::width() const { return high - low; }

    __host__ __device__ Intervalf operator+(Intervalf a, Intervalf b)
    {
        return {fl::addRoundDown(a.low, b.low), fl::addRoundUp(a.high, b.high)};
    }

    __host__ __device__ Intervalf operator-(Intervalf a, Intervalf b)
    {
        return {fl::subRoundDown(a.low, b.low), fl::subRoundUp(a.high, b.high)};
    }

    __host__ __device__ Intervalf operator*(Intervalf a, Intervalf b)
    {
        float lp[4] = {fl::mulRoundDown(a.low, b.low),
                       fl::mulRoundDown(a.high, b.low),
                       fl::mulRoundDown(a.low, b.high),
                       fl::mulRoundDown(a.high, b.high)};
        float hp[4] = {fl::mulRoundUp(a.low, b.low),
                       fl::mulRoundUp(a.high, b.low),
                       fl::mulRoundUp(a.low, b.high),
                       fl::mulRoundUp(a.high, b.high)};
        return {std::min({lp[0], lp[1], lp[2], lp[3]}), std::max({hp[0], hp[1], hp[2], hp[3]})};
    }

    __host__ __device__ Intervalf operator/(Intervalf a, Intervalf b)
    {
        // if the interval of the divisor contains zero, return the whole extended real number line
        if (b.low < 0.f && b.high > 0.f)
            return {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};

        float lowQuot[4]  = {fl::divRoundDown(a.low, b.low),
                             fl::divRoundDown(a.high, b.low),
                             fl::divRoundDown(a.low, b.high),
                             fl::divRoundDown(a.high, b.high)};
        float highQuot[4] = {fl::divRoundUp(a.low, b.low),
                             fl::divRoundUp(a.high, b.low),
                             fl::divRoundUp(a.low, b.high),
                             fl::divRoundUp(a.high, b.high)};
        return {std::min({lowQuot[0], lowQuot[1], lowQuot[2], lowQuot[3]}),
                std::max({highQuot[0], highQuot[1], highQuot[2], highQuot[3]})};
    }
} // namespace dmt