#include "cudautils/cudautils-float.cuh"

// GLM and Eigen
#include "cudautils-include-glm.cuh"

namespace dmt::fl {
    __host__ __device__ float abs(float x)
    {
#if defined(__CUDA_ARCH__)
        return ::fabsf(x);
#elif defined(DMT_ARCH_X86_64)
        return _mm_cvtss_f32(_mm_andnot_ps(_mm_set_ss(-0.f), _mm_set_ss(x)));
#else
    #error "CPU Architecture not supported"
#endif
    }

    __host__ __device__ float rsqrt(float x)
    {
#if defined(__CUDA_ARCH__)
        return __fsqrt_rn(x);
#elif defined(DMT_ARCH_X86_64)

        return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
#else
    #error "CPU Architecture not supported"
#endif
    }

    __host__ __device__ float rcp(float x)
    {
        float ret = x;
#if !defined(DMT_SKIP_FLOAT_TESTS)
    #if defined(__CUDA_ARCH__)
        if (glm::epsilonEqual(x, 0.f, 1e-7f))
    #else
        if (glm::epsilonEqual(x, 0.f, eqTol()))
    #endif
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

    __host__ __device__ bool nearZero(float x, float tol)
    {
#if defined(__CUDA_ARCH__)
        return glm::epsilonEqual(x, 0.f, 1e-7f);
#else
        return glm::epsilonEqual(x, 0.f, tol);
#endif
    }

    __host__ __device__ bool near(float x, float y)
    {
#if defined(__CUDA_ARCH__)
        return glm::epsilonEqual(x, 0.f, 1e-7f);
#else
        return glm::epsilonEqual(x, y, eqTol());
#endif
    }

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
        float lp0 = fl::mulRoundDown(a.low, b.low);
        float lp1 = fl::mulRoundDown(a.high, b.low);
        float lp2 = fl::mulRoundDown(a.low, b.high);
        float lp3 = fl::mulRoundDown(a.high, b.high);

        float hp0 = fl::mulRoundUp(a.low, b.low);
        float hp1 = fl::mulRoundUp(a.high, b.low);
        float hp2 = fl::mulRoundUp(a.low, b.high);
        float hp3 = fl::mulRoundUp(a.high, b.high);

        float low = lp0;
        if (lp1 < low)
            low = lp1;
        if (lp2 < low)
            low = lp2;
        if (lp3 < low)
            low = lp3;

        float high = hp0;
        if (hp1 > high)
            high = hp1;
        if (hp2 > high)
            high = hp2;
        if (hp3 > high)
            high = hp3;

        return {low, high};
    }

    __host__ __device__ Intervalf operator/(Intervalf a, Intervalf b)
    {
        // if divisor interval contains zero → full real line
        if (b.low < 0.f && b.high > 0.f)
            return {-__builtin_inff(), __builtin_inff()};

        float lq0 = fl::divRoundDown(a.low, b.low);
        float lq1 = fl::divRoundDown(a.high, b.low);
        float lq2 = fl::divRoundDown(a.low, b.high);
        float lq3 = fl::divRoundDown(a.high, b.high);

        float hq0 = fl::divRoundUp(a.low, b.low);
        float hq1 = fl::divRoundUp(a.high, b.low);
        float hq2 = fl::divRoundUp(a.low, b.high);
        float hq3 = fl::divRoundUp(a.high, b.high);

        float low = lq0;
        if (lq1 < low)
            low = lq1;
        if (lq2 < low)
            low = lq2;
        if (lq3 < low)
            low = lq3;

        float high = hq0;
        if (hq1 > high)
            high = hq1;
        if (hq2 > high)
            high = hq2;
        if (hq3 > high)
            high = hq3;

        return {low, high};
    }

} // namespace dmt
