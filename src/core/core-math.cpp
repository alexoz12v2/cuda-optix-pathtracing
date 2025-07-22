#include "core-math.h"

namespace dmt::arch {
    float hmin_ps(__m128 v)
    {
        __m128 shuf = _mm_movehdup_ps(v); // (v1,v1,v3,v3)
        __m128 mins = _mm_min_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, mins); // (v2,v3)
        mins        = _mm_min_ss(mins, shuf);
        return _mm_cvtss_f32(mins);
    }

    float hmax_ps(__m128 v)
    {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 maxs = _mm_max_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, maxs);
        maxs        = _mm_max_ss(maxs, shuf);
        return _mm_cvtss_f32(maxs);
    }

    float hmin_ps(__m256 v)
    {
        __m128 low  = _mm256_castps256_ps128(v);   // lower 128
        __m128 high = _mm256_extractf128_ps(v, 1); // upper 128
        __m128 min1 = _mm_min_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(min1);
        __m128 min2 = _mm_min_ps(min1, shuf);
        shuf        = _mm_movehl_ps(shuf, min2);
        min2        = _mm_min_ss(min2, shuf);
        return _mm_cvtss_f32(min2);
    }

    float hmax_ps(__m256 v)
    {
        __m128 low  = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 max1 = _mm_max_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(max1);
        __m128 max2 = _mm_max_ps(max1, shuf);
        shuf        = _mm_movehl_ps(shuf, max2);
        max2        = _mm_max_ss(max2, shuf);
        return _mm_cvtss_f32(max2);
    }
} // namespace dmt::arch
