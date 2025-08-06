#include "core-math.h"

#include "platform/platform-memory.h"

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

    void transpose3x2(float const* src, float* x, float* y, float* z)
    {
        // Assumes src is [x0, y0, z0, x1, y1, z1]
        x[0] = src[0];
        x[1] = src[3];
        y[0] = src[1];
        y[1] = src[4];
        z[0] = src[2];
        z[1] = src[5];
    }

    // TODO see if you can do it better with less shuffles using _mm_unpack(lo/hi)_ps, _mm_move(lh/hl)_ps
    void transpose3x4(float const* src, float* x, float* y, float* z)
    {
        // src layout: [x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3]
        __m128 a = _mm_loadu_ps(src);     // x0 y0 z0 x1
        __m128 b = _mm_loadu_ps(src + 4); // y1 z1 x2 y2
        __m128 c = _mm_loadu_ps(src + 8); // z2 x3 y3 z3

        // x
        __m128 x23 = _mm_shuffle_ps(b, c, _MM_SHUFFLE(2, 2, 1, 1));   // x2 x2 x3 x3
        __m128 xv  = _mm_shuffle_ps(a, x23, _MM_SHUFFLE(0, 3, 0, 2)); // x0 x1 x2 x3

        // y
        __m128 yz01 = _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 2, 0, 1));      // y0 z0 y1 z1
        __m128 y23  = _mm_shuffle_ps(b, c, _MM_SHUFFLE(3, 3, 2, 2));      // y2 y2 y3 y3
        __m128 yv   = _mm_shuffle_ps(yz01, y23, _MM_SHUFFLE(0, 2, 0, 2)); // y0 y1 y2 y3

        // z
        __m128 zv = _mm_shuffle_ps(yz01, c, _MM_SHUFFLE(1, 3, 0, 3)); // z0 z1 z2 z3

        // store
        _mm_storeu_ps(x, xv);
        _mm_storeu_ps(y, yv);
        _mm_storeu_ps(z, zv);
    }

    // TODO translate the following:
    //transpose3x8(float const*, float*, float*, float*):
    //    vmovss  xmm0, DWORD PTR [rdi+76]
    //    vinsertps       xmm1, xmm0, DWORD PTR [rdi+88], 0x10
    //    vmovss  xmm0, DWORD PTR [rdi+52]
    //    vinsertps       xmm0, xmm0, DWORD PTR [rdi+64], 0x10
    //    vmovlhps        xmm0, xmm0, xmm1
    //    vmovss  xmm1, DWORD PTR [rdi+28]
    //    vinsertps       xmm2, xmm1, DWORD PTR [rdi+40], 0x10
    //    vmovss  xmm1, DWORD PTR [rdi+4]
    //    vinsertps       xmm1, xmm1, DWORD PTR [rdi+16], 0x10
    //    vmovlhps        xmm1, xmm1, xmm2
    //    vmovss  xmm2, DWORD PTR [rdi+56]
    //    vinsertf128     ymm1, ymm1, xmm0, 0x1
    //    vinsertps       xmm2, xmm2, DWORD PTR [rdi+68], 0x10
    //    vmovss  xmm0, DWORD PTR [rdi+80]
    //    vinsertps       xmm0, xmm0, DWORD PTR [rdi+92], 0x10
    //    vmovlhps        xmm2, xmm2, xmm0
    //    vmovss  xmm0, DWORD PTR [rdi+32]
    //    vinsertps       xmm3, xmm0, DWORD PTR [rdi+44], 0x10
    //    vmovss  xmm0, DWORD PTR [rdi+8]
    //    vinsertps       xmm0, xmm0, DWORD PTR [rdi+20], 0x10
    //    vmovlhps        xmm0, xmm0, xmm3
    //    vmovss  xmm3, DWORD PTR [rdi+48]
    //    vinsertf128     ymm0, ymm0, xmm2, 0x1
    //    vinsertps       xmm3, xmm3, DWORD PTR [rdi+60], 0x10
    //    vmovss  xmm2, DWORD PTR [rdi+72]
    //    vinsertps       xmm2, xmm2, DWORD PTR [rdi+84], 0x10
    //    vmovlhps        xmm3, xmm3, xmm2
    //    vmovss  xmm2, DWORD PTR [rdi+24]
    //    vinsertps       xmm4, xmm2, DWORD PTR [rdi+36], 0x10
    //    vmovss  xmm2, DWORD PTR [rdi]
    //    vinsertps       xmm2, xmm2, DWORD PTR [rdi+12], 0x10
    //    vmovlhps        xmm2, xmm2, xmm4
    //    vinsertf128     ymm2, ymm2, xmm3, 0x1
    //    vmovups YMMWORD PTR [rsi], ymm2
    //    vmovups YMMWORD PTR [rdx], ymm1
    //    vmovups YMMWORD PTR [rcx], ymm0
    //    vzeroupper
    //    ret
    void transpose3x8(float const* src, float* x, float* y, float* z)
    {
        // src layout: [x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5 y5 z5 x6 y6 z6 x7 y7 z7]
        __m256 a = _mm256_loadu_ps(src);      // x0 y0 z0 x1 y1 z1 x2 y2
        __m256 b = _mm256_loadu_ps(src + 8);  // z2 x3 y3 z3 x4 y4 z4 x5
        __m256 c = _mm256_loadu_ps(src + 16); // y5 z5 x6 y6 z6 x7 y7 z7

        // vperm2f128 -> y1 z1 x2 y2 z2 x3 y3 z3 | (a, b, 0b0010'0001)
        // vpermq     -> z0 x1 x2 y2 ** ** ** ** | (a, 0bxx'xx'11'01)
        // vpermq     -> ** ** z2 x3 x4 y4 z4 x5 | (b, 0b11'10'00'xx)

        // src: 24 floats = 8 * (x, y, z)
        __m256 v0 = _mm256_loadu_ps(src);      // [x0 y0 z0 x1 y1 z1 x2 y2]
        __m256 v1 = _mm256_loadu_ps(src + 8);  // [z2 x3 y3 z3 x4 y4 z4 x5]
        __m256 v2 = _mm256_loadu_ps(src + 16); // [y5 z5 x6 y6 z6 x7 y7 z7]

        // Step 1: shuffle & unpack into partial x/y/z channels
        // Gather X
        __m256 x_part1 = _mm256_set_ps(src[21], src[18], src[15], src[12], src[9], src[6], src[3], src[0]);
        // Gather Y
        __m256 y_part1 = _mm256_set_ps(src[22], src[19], src[16], src[13], src[10], src[7], src[4], src[1]);
        // Gather Z
        __m256 z_part1 = _mm256_set_ps(src[23], src[20], src[17], src[14], src[11], src[8], src[5], src[2]);

        _mm256_storeu_ps(x, x_part1);
        _mm256_storeu_ps(y, y_part1);
        _mm256_storeu_ps(z, z_part1);
    }

    void transpose3xN(float const* src, float* x, float* y, float* z, size_t N)
    {
        size_t i = 0;
        for (; i + 7 < N; i += 8)
            transpose3x8(src + 3 * i, x + i, y + i, z + i);

        // residual
        for (; i < N; ++i)
        {
            x[i] = src[3 * i + 0];
            y[i] = src[3 * i + 1];
            z[i] = src[3 * i + 2];
        }
    }
} // namespace dmt::arch

namespace dmt::color {
    RGB rgbFromHsv(Point3f hsv)
    {
        auto [h, s, v] = hsv;

        h = std::fmodf(h, 1.0f);
        if (h < 0.0f)
            h += 1.0f;

        float c = v * s;
        float x = c * (1 - std::fabs(std::fmodf(h * 6.0f, 2.0f) - 1));
        float m = v - c;

        float r1, g1, b1;
        // clang-format off
        if      (h < 1.0f / 6.0f) { r1 = c; g1 = x; b1 = 0; }
        else if (h < 2.0f / 6.0f) { r1 = x; g1 = c; b1 = 0; }
        else if (h < 3.0f / 6.0f) { r1 = 0; g1 = c; b1 = x; }
        else if (h < 4.0f / 6.0f) { r1 = 0; g1 = x; b1 = c; }
        else if (h < 5.0f / 6.0f) { r1 = x; g1 = 0; b1 = c; }
        else                      { r1 = c; g1 = 0; b1 = x; }
        // clang-format on

        return {r1 + m, g1 + m, b1 + m};
    }

    Point3f hsvFromRgb(RGB rgb)
    {
        static constexpr float _60Deg = 60.f / 360.f;

        Vector3f    v{rgb.r, rgb.g, rgb.b};
        float const value            = maxComponent(v);
        float const valueMinusChroma = minComponent(v);

        float const chroma    = value - valueMinusChroma;
        float const invChroma = fl::rcp(chroma);

        float hue = 0.f;
        if (fl::nearZero(chroma))
            hue = 0.f;
        else if (value == rgb.r)
            hue = _60Deg * std::fmodf((rgb.g - rgb.b) * invChroma, 6.f);
        else if (value == rgb.g)
            hue = _60Deg * (rgb.b - rgb.r) * invChroma + 2.f;
        else // value == rgb.b
            hue = _60Deg * (rgb.r - rgb.g) * invChroma + 4.f;

        float saturation = 0.f;
        if (fl::nearZero(value))
            saturation = 0.f;
        else
            saturation = chroma / value;

        return {hue, saturation, value};
    }
} // namespace dmt::color

namespace dmt {
    static uint16_t encodeOcta(float f)
    {
        static constexpr float max = std::numeric_limits<uint16_t>::max();
        return static_cast<uint16_t>(fl::round(fl::clamp01((f + 1) / 2) * max));
    }

    OctahedralNorm octaFromNorm(Normal3f n)
    {
        // 1: planar projection
        Vector3f projected = n.asVec() / (fl::abs(n.x) + fl::abs(n.y) + fl::abs(n.z));

        // 2: outfold the downward faces to the external triangles of the quad
        if (projected.z < 0.f)
        {
            float const ox = projected.x;
            float const oy = projected.y;
            projected.x    = (1.f - fl::abs(oy)) * fl::sign(ox);
            projected.y    = (1.f - fl::abs(ox)) * fl::sign(oy);
        }
        // 3: mapping into [0,1]
        return {.x = encodeOcta(projected.x), .y = encodeOcta(projected.y)};
    }


    Normal3f normFromOcta(OctahedralNorm o)
    {
        static constexpr float max = std::numeric_limits<uint16_t>::max();

        // 1: convert back to [-1, 1] range
        Vector2f const f{static_cast<float>(o.x) / max * 2.f - 1.f, static_cast<float>(o.y) / max * 2.f - 1.f};

        // 2: reconstruct z component
        Vector3f n(f.x, f.y, 1.f - fl::abs(f.x) - fl::abs(f.y));

        // 3: fold back the lower hemisphere
        if (n.z < 0.f)
        {
            float const ox = n.x;
            float const oy = n.y;

            n.x = (1.f - fl::abs(oy)) * fl::sign(ox);
            n.y = (1.f - fl::abs(ox)) * fl::sign(oy);
        }

        // 4: normalize
        return Normal3f{n};
    }

    void extractAffineTransform(Matrix4f const& m, float affineTransform[12])
    {
        affineTransform[0] = m.m[0]; // m(0,0)
        affineTransform[1] = m.m[1]; // m(0,1)
        affineTransform[2] = m.m[2]; // m(0,2)

        affineTransform[3] = m.m[4]; // m(1,0)
        affineTransform[4] = m.m[5]; // m(1,1)
        affineTransform[5] = m.m[6]; // m(1,2)

        affineTransform[6] = m.m[8];  // m(2,0)
        affineTransform[7] = m.m[9];  // m(2,1)
        affineTransform[8] = m.m[10]; // m(2,2)

        affineTransform[9]  = m.m[12]; // m(3,0) -> translation x
        affineTransform[10] = m.m[13]; // m(3,1) -> translation y
        affineTransform[11] = m.m[14]; // m(3,2) -> translation z
    }

    Matrix4f matrixFromAffine(float const affineTransform[12])
    {
        // clang-format off
        Matrix4f const m {
            affineTransform[0], affineTransform[1],  affineTransform[2],  0,
            affineTransform[3], affineTransform[4],  affineTransform[5],  0,
            affineTransform[6], affineTransform[7],  affineTransform[8],  0,
            affineTransform[9], affineTransform[10], affineTransform[11], 1
        };
        // clang-format on
        return m;
    }

    Transform transformFromAffine(float const affineTransform[12])
    {
        return Transform{matrixFromAffine(affineTransform)};
    }


    float lookupTableRead(float const* table, float x, int32_t size)
    {
        x = fl::clamp01(x) * (size - 1);

        int32_t const index  = fminf(static_cast<int32_t>(x), size - 1);
        int32_t const nIndex = fminf(index + 1, size - 1);
        float const   t      = x - index;

        // lerp formula
        float const data0 = table[index];
        if (t == 0.f)
            return data0;

        float const data1 = table[nIndex];
        return (1.f - t) * data0 + t * data1;
    }

    float lookupTableRead2D(float const* table, float x, float y, int32_t sizex, int32_t sizey)
    {
        y = fl::clamp01(x) * (sizey - 1);

        int32_t const index  = fminf(static_cast<int32_t>(y), sizey - 1);
        int32_t const nIndex = fminf(index + 1, sizey - 1);
        float const   t      = y - index;

        // bilinear interp formula
        float const data0 = lookupTableRead(table + sizex * index, x, sizex);
        if (t == 0.f)
            return data0;

        float const data1 = lookupTableRead(table + sizex * nIndex, x, sizex);
        return (1.f - t) * data0 + t * data1;
    }

    PiecewiseConstant1D::PiecewiseConstant1D(std::span<float const> func, float min, float max, std::pmr::memory_resource* memory) :
    m_buffer{makeUniqueRef<float[]>(memory, func.size() << 1)},
    m_funcCount(static_cast<decltype(m_funcCount)>(func.size())),
    m_min(min),
    m_max(max)
    {
        assert(isPOT(func.size()) && "PiecewiseConstant1D requires its source sampled function to have POT samples");
        assert(func.size() == m_funcCount && "narrowing conversion of size lost values");

        // First step: copy absolute value of function
        float const* fPtr      = func.data();
        float*       fDest     = m_buffer.get();
        uint32_t     remaining = m_funcCount;

        __m256 const nzero8 = _mm256_set1_ps(-0.f);
        __m128 const nzero4 = _mm_set1_ps(-0.f);
        while (remaining >= 8)
        { // store abs
            _mm256_storeu_ps(fDest, _mm256_andnot_ps(nzero8, _mm256_loadu_ps(fPtr)));
            fPtr += 8;
            fDest += 8;
            remaining -= 8;
        }

        while (remaining >= 4)
        {
            _mm_storeu_ps(fDest, _mm_andnot_ps(nzero4, _mm_loadu_ps(fPtr)));
            fPtr += 4;
            fDest += 4;
            remaining -= 4;
        }

        while (remaining != 0)
        {
            *fDest++ = *fPtr++;
            --remaining;
        }

        // Second Step: CDF Computation https://en.algorithmica.org/hpc/algorithms/prefix/
        // split the array into blocks of 8 (remaining later)
        // cdf[0] = 0;
        // for (size_t i = 1; i < n + 1; ++i)
        //     cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
        uint32_t const numBlocksTimes8 = m_funcCount & ~0x7u;
        fPtr                           = func.data();
        float*       cdf               = m_buffer.get() + m_funcCount;
        float        carry             = 0.f;
        __m256 const normalizeFac      = _mm256_set1_ps((m_max - m_min) / m_funcCount);

        for (uint32_t i = 0; i < numBlocksTimes8 >> 3u; ++i)
        {
            __m256 x = _mm256_mul_ps(_mm256_loadu_ps(fPtr), normalizeFac);

            // In-lane prefix sum (lane 0: [0–3], lane 1: [4–7])
            __m256 t;

            t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, t);

            t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, t);

            // Extract lane 0 sum (last element of lane 0)
            __m128 low       = _mm256_castps256_ps128(x);
            float  lane0_sum = _mm_cvtss_f32(_mm_shuffle_ps(low, low, _MM_SHUFFLE(3, 3, 3, 3)));

            // Broadcast lane0_sum to a vector
            __m256 lane0_sum_vec = _mm256_set1_ps(lane0_sum);

            // Create mask to zero out lane 0, keep lane 1
            __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0, -1, -1, -1, -1));

            // Add lane0_sum only to lane 1 elements
            x = _mm256_blendv_ps(x, _mm256_add_ps(x, lane0_sum_vec), mask);

            // Step 2: Add carry from previous block
            __m256 carryVec = _mm256_set1_ps(carry);
            x               = _mm256_add_ps(x, carryVec);

            _mm256_storeu_ps(cdf, x); // store result

            // Step 3: Extract last value from x to use as carry for next block
            // The last element in x is the total sum of this block
            // To get it, extract the high 128-bit lane, then extract element 3

            __m128 high = _mm256_extractf128_ps(x, 1);                              // get elements 4–7
            float  last = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0b11'11'11'11)); // get element 7

            carry = last;
            fPtr += 8;
            cdf += 8;
        }
        // handle remainder with carry and scalar ops
        remaining = m_funcCount - numBlocksTimes8;
        while (remaining != 0)
        {
            *cdf = *(cdf - 1) + *fPtr;
            ++cdf;
            ++fPtr;
            --remaining;
        }

        // step 3: normalize CDF
        //   funcInt = cdf[n];
        //   if (funcInt == 0) for (size_t i = 1; i < n + 1; ++i) cdf[i] = Float(i) / Float(n);
        //   else              for (size_t i = 1; i < n + 1; ++i) cdf[i] /= funcInt;
        m_integral        = CDF_().back();
        float const  fac  = dmt::fl::rcp(dmt::fl::nearZero(m_integral) ? m_funcCount : m_integral);
        __m256 const vFac = _mm256_set1_ps(fac);

        cdf       = m_buffer.get() + m_funcCount;
        remaining = m_funcCount - numBlocksTimes8;

        if (dmt::fl::nearZero(m_integral))
        {
            alignas(alignof(__m256)) float base[8]{0, 1, 2, 3, 4, 5, 6, 7};

            __m256       vNum    = _mm256_load_ps(base);
            __m256 const vOffset = _mm256_set1_ps(8.f);
            for (uint32_t i = 0; i < numBlocksTimes8; ++i)
            {
                _mm256_storeu_ps(cdf, _mm256_mul_ps(vNum, vFac));
                vNum = _mm256_add_ps(vNum, vOffset);
                cdf += 8;
            }

            // Handle remainder scalars
            for (size_t i = 0; i < remaining; ++i)
                cdf[i] = static_cast<float>(numBlocksTimes8 + i) * fac;
        }
        else
        {
            for (uint32_t i = 0; i < numBlocksTimes8 >> 3; ++i)
            {
                _mm256_storeu_ps(cdf, _mm256_mul_ps(_mm256_loadu_ps(cdf), vFac));
                cdf += 8;
            }

            // Handle remainder scalars
            for (size_t i = 0; i < remaining; ++i)
                cdf[i] *= fac;
        }
    }

    float PiecewiseConstant1D::integral() const { return m_integral; }

    uint32_t PiecewiseConstant1D::size() const { return m_funcCount; }

    float PiecewiseConstant1D::invert(float x) const
    {
        if (x < m_min || x > m_max)
            return std::numeric_limits<float>::quiet_NaN();
        float const   c      = (x - m_min) / (m_max - m_min) * m_funcCount;
        int32_t const offset = dmt::fl::clamp(static_cast<int32_t>(c), 0, static_cast<int32_t>(m_funcCount - 1));

        float const delta = c - offset;
        auto        cdf   = CDF();
        return dmt::fl::lerp(delta, cdf[offset], cdf[offset + 1]);
    }

    static int32_t findIntervalLessThan(int32_t sz, std::span<float const> cdf, float u)
    {
        int32_t size = sz - 2, first = 1;
        while (size > 0)
        {
            // Evaluate predicate at midpoint and update _first_ and _size_
            size_t half = (size_t)size >> 1, middle = first + half;
            bool   predResult = cdf[middle] <= u;
            first             = predResult ? middle + 1 : first;
            size              = predResult ? size - (half + 1) : half;
        }
        return std::clamp(first - 1, 0, sz - 2);
    }

    float PiecewiseConstant1D::sample(float u, float* pdf, int32_t* offset) const
    {
        auto cdf  = CDF();
        auto func = absFunc();

        int32_t const off = findIntervalLessThan(m_funcCount, cdf, u);
        if (offset)
            *offset = off;

        // compute offset along CDF segment (linear interp formula)
        float du = u - cdf[off];
        if (cdf[off + 1] - cdf[off] > 0)
            du /= cdf[off + 1] - cdf[off];

        if (pdf)
            *pdf = m_integral > 0 ? func[off] / m_integral : 0;

        return dmt::fl::lerp((off + du) / m_funcCount, m_min, m_max);
    }

    // PiecewiseConstant2D
    static std::pmr::vector<PiecewiseConstant1D> makePConditional(dstd::Array2D<float> const& data,
                                                                  Bounds2f                    domain,
                                                                  std::pmr::memory_resource*  memory)
    {
        std::pmr::vector<PiecewiseConstant1D> nrvo{memory};
        nrvo.reserve(data.ySize());
        for (uint32_t i = 0; i < data.ySize(); ++i)
        {
            nrvo.emplace_back(data.rowSpan(i), domain.pMin[0], domain.pMax[0], memory);
        }

        return nrvo;
    }

    static std::pmr::vector<float> makeMarginalFunc(std::span<PiecewiseConstant1D> pConditionalV,
                                                    std::pmr::memory_resource*     memory)
    {
        std::pmr::vector<float> marginalFunc{pConditionalV.size(), memory};
        for (uint32_t i = 0; i < marginalFunc.size(); ++i)
            marginalFunc[i] = pConditionalV[i].integral();
        return marginalFunc;
    }

    PiecewiseConstant2D::PiecewiseConstant2D(dstd::Array2D<float> const& data,
                                             Bounds2f                    domain,
                                             std::pmr::memory_resource*  memory,
                                             std::pmr::memory_resource*  temp) :
    m_domain(domain),
    m_pConditionalV(makePConditional(data, domain, memory)),
    m_pMarginalV(makeMarginalFunc(m_pConditionalV, temp), domain.pMin[1], domain.pMax[1], memory)
    {
        assert(isPOT(data.xSize()) && isPOT(data.ySize()) && "We want powers of two");
    }

    Point2f PiecewiseConstant2D::sample(Point2f u, float* pdf, Point2i* offset) const
    {
        float   pdfs[2];
        Point2i uv;
        float   d1 = m_pMarginalV.sample(u[1], &pdfs[1], &uv[1]);
        float   d0 = m_pConditionalV[uv[1]].sample(u[0], &pdfs[0], &uv[0]);

        if (pdf) // p(x,y) = p(x|y) * p(y)
            *pdf = pdfs[0] * pdfs[1];
        if (offset)
            *offset = uv;

        assert(fl::abs(d0) <= 1.f && fl::abs(d1) <= 1.f);
        return {{d0, d1}};
    }

    float PiecewiseConstant2D::pdf(Point2f pr) const
    {
        // take the chosen conditional pdf, take its generating, unnormalized function, and divide it by marginal integral.
        // this is equivalent to doing p(x,y) = p(x|y) * p(y)
        // project onto domain and take indices to marginal and conditional
        Point2f const p{{m_domain.offset(pr)}};
        int32_t const iu = clamp(static_cast<int32_t>(p[0] * m_pConditionalV[0].size()),
                                 0,
                                 static_cast<int32_t>(m_pConditionalV[0].size() - 1));
        int32_t const iv = clamp(static_cast<int32_t>(p[1] * m_pMarginalV.size()),
                                 0,
                                 static_cast<int32_t>(m_pMarginalV.size() - 1));
        // absFunc[iv] stores p(x|y=at(iv)) * p(y=at(iv))
        // hence absFunc[iv][iu], is the unnormalized p(x|y) * p(y) evaluated for y=at(iv), x=at(iu)
        // to normalize that, we need to divide by total integral, which is equal to the integral of the marginal
        return m_pConditionalV[iv].absFunc()[iu] / m_pMarginalV.integral();
    }

    Point2f PiecewiseConstant2D::invert(Point2f p) const
    {
        static Point2f const outside{{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()}};
        // first invert the marginal
        float const mInv = m_pMarginalV.invert(p[1]);
        if (fl::isNaN(mInv))
            return outside;

        float const percentageOverMarginalDomain = (p[1] - m_domain.pMin[1]) / (m_domain.pMax[1] - m_domain.pMin[1]);
        if (percentageOverMarginalDomain < 0.f || percentageOverMarginalDomain > 1.f)
            return outside;

        int32_t const condIndex = clamp(static_cast<int32_t>(percentageOverMarginalDomain * m_pConditionalV.size()),
                                        0,
                                        static_cast<int32_t>(m_pConditionalV.size() - 1));
        float const   cInv      = m_pConditionalV[condIndex].invert(p[0]);
        if (fl::isNaN(cInv))
            return outside;

        return {{cInv, mInv}};
    }

    Bounds2f PiecewiseConstant2D::domain() const { return m_domain; }

    Point2i PiecewiseConstant2D::resolution() const
    {
        return {{static_cast<int32_t>(m_pConditionalV[0].size()), static_cast<int32_t>(m_pMarginalV.size())}};
    }

    float PiecewiseConstant2D::integral() const { return m_pMarginalV.integral(); }
} // namespace dmt

namespace dmt::transforms {
    Transform DMT_FASTCALL persp(float fovRadians, float aspectRatio, float near, float far)
    {
        float const focalLen = fl::rcp(tan(fovRadians * 0.5f));

        float const x = focalLen / aspectRatio;
        float const y = -focalLen;

        // OpenGL style [z -> -1, 1]
        float const a = near / (far - near);
        float const b = far * a;

        // Direct3D style [z -> 0, 1] (PBRT)
        // float const a = far / (far - near);
        // float const b = -near * a;

        // clang-format off
        Matrix4f const m{{
            x, 0, 0,  0, // first column
            0, y, 0,  0, // second column
            0, 0, a, -1, // third column
            0, 0, b,  0  // fourth column
        }};
        // clang-format on
        return dmt::Transform{m};
    }

    Transform DMT_FASTCALL scale(Vector3f s)
    {
        // clang-format off
        Matrix4f const m{{
            s[0], 0,    0,    0, // first column
            0,    s[1], 0,    0, // second column
            0,    0,    s[2], 0, // third column
            0,    0,    0,    1  // fourth column
        }};
        // clang-format on
        return dmt::Transform{m};
    }

    Transform DMT_FASTCALL translate(Vector3f v)
    {
        // clang-format off
        Matrix4f const m{{
            1,    0,    0,    0, // first column
            0,    1,    0,    0, // second column
            0,    0,    1,    0, // third column
            v[0], v[1], v[2], 1  // fourth column
        }};
        // clang-format on
        return dmt::Transform{m};
    }

    Transform DMT_FASTCALL cameraWorldFromCamera(Normal3f cameraDirection)
    {
        Normal3f tmpUp{{0, 0, 1}};
        if (absDot(cameraDirection, tmpUp) > 0.99f)
            tmpUp = {{0, 1, 0}}; // handle gimbal lock

        // orthonormal basis
        Normal3f const right = normalFrom(cross(tmpUp, cameraDirection)); // world X
        Normal3f const up    = normalFrom(cross(cameraDirection, right)); // world Z

        // clang-format off
        Matrix4f const m{{
            right.x,  up.x,    cameraDirection.x,  0,
            right.y,  up.y,    cameraDirection.y,  0,
            right.z,  up.z,    cameraDirection.z,  0,
            0,        0,       0,                  1
        }};
        // clang-format on

        return dmt::Transform{m};
    }

    // | Rx Ux Fx Tx |
    // | Ry Uy Fy Ty |
    // | Rz Uz Fz Tz |
    // | 0  0  0  1  |
    Transform DMT_FASTCALL worldFromCamera(Normal3f cameraDirection, Point3f cameraPosition)
    {
        Vector3f const forward = cameraDirection; // +Z in camera space
        Vector3f const worldUp{0.0f, 0.0f, 1.0f}; // World up (0, 0, +Z)

        // Compute right (X) and up (Y) vectors for the camera frame (left handed system)
        Vector3f right = cross(forward, worldUp); // +X in camera space
        Vector3f up    = cross(right, forward);   // +Y in camera space
        right          = normalize(right);
        up             = normalize(up);

        // Column-major matrix for worldFromCamera
        // clang-format off
        Matrix4f const m{{
            right.x,   right.y,   right.z,   0.0f, // Column 0: right
            up.x,      up.y,      up.z,      0.0f, // Column 1: up
            forward.x, forward.y, forward.z, 0.0f, // Column 2: forward
            cameraPosition.x, cameraPosition.y, cameraPosition.z, 1.0f // Column 3: position
        }};
        // clang-format on
        assert(determinant(m) < 0 &&
               "left handed camera space -> right handed world space"
               " should swap handedness");

        return Transform{m};
    }

    Transform DMT_FASTCALL cameraFromRaster_Perspective(float focalLength, float sensorHeight, uint32_t xRes, uint32_t yRes)
    {
        float const aspectRatio = static_cast<float>(xRes) / static_cast<float>(yRes);
        float const halfHeight  = 0.5f * sensorHeight;
        float const halfWidth   = halfHeight * aspectRatio;

        float const pixelSizeX = 2.0f * halfWidth / static_cast<float>(xRes);
        float const pixelSizeY = 2.0f * halfHeight / static_cast<float>(yRes);

        float const tx = -halfWidth + 0.5f * pixelSizeX;
        float const ty = halfHeight - 0.5f * pixelSizeY;

        Matrix4f const m{
            {pixelSizeX, 0.0f, 0.0f, 0.0f, 0.0f, -pixelSizeY, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, tx, ty, focalLength, 1.0f}};

        return dmt::Transform{m};
    }
} // namespace dmt::transforms
