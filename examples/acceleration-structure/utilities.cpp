#include "utilities.h"

#include "platform/platform-context.h"

#if !defined(DMT_ARCH_X86_64)
    #error "what"
#endif

#include <immintrin.h>

namespace dmt {
    bool DMT_FASTCALL slabTest(Point3f rayOrigin, Vector3f rayDirection, Bounds3f const& box, float* outTmin, float* outTmax)
    {
        float tmin = -std::numeric_limits<float>::infinity();
        float tmax = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; ++i)
        {
            float invD = 1.0f / rayDirection[i];
            float t0   = (box.pMin[i] - rayOrigin[i]) * invD;
            float t1   = (box.pMax[i] - rayOrigin[i]) * invD;

            if (invD < 0.0f)
                std::swap(t0, t1);

            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);

            if (tmax < tmin)
                return false; // No intersection
        }

        if (outTmin)
            *outTmin = tmin;
        if (outTmax)
            *outTmax = tmax;

        return true; // Intersection occurred
    }

    std::pmr::vector<UniqueRef<Primitive>> makeSinglePrimitivesFromTriangles(std::span<TriangleData const> tris,
                                                                             std::pmr::memory_resource*    memory)
    {
        std::pmr::vector<UniqueRef<Primitive>> out(memory);
        for (uint64_t i = 0; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    std::pmr::vector<UniqueRef<Primitive>> makePrimitivesFromTriangles(std::span<TriangleData const> tris,
                                                                       std::pmr::memory_resource*    memory)
    {
        std::pmr::vector<UniqueRef<Primitive>> out(memory);
        size_t                                 i = 0;

        // Pass 1: Triangles8
        for (; i + 8 <= tris.size(); i += 8)
        {
            Triangles8 group{};
            for (int j = 0; j < 8; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(makeUniqueRef<Triangles8>(memory, std::move(group)));
        }

        // Pass 2: Triangles4
        for (; i + 4 <= tris.size(); i += 4)
        {
            Triangles4 group{};
            for (int j = 0; j < 4; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(makeUniqueRef<Triangles4>(memory, std::move(group)));
        }

        // Pass 3: Triangles2
        for (; i + 2 <= tris.size(); i += 2)
        {
            Triangles2 group{};
            for (int j = 0; j < 2; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(makeUniqueRef<Triangles2>(memory, std::move(group)));
        }

        // Pass 4: Individual Triangle
        for (; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    uint32_t morton3D(float x, float y, float z)
    {
        constexpr auto expandBits = [](uint32_t v) -> uint32_t {
            // Expands 10 bits into 30 bits by inserting 2 zeros between each bit
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };
        // Assumes x, y, z are ∈ [0, 1]
        x = std::clamp(x * 1024.0f, 0.0f, 1023.0f);
        y = std::clamp(y * 1024.0f, 0.0f, 1023.0f);
        z = std::clamp(z * 1024.0f, 0.0f, 1023.0f);

        uint32_t xx = expandBits(static_cast<uint32_t>(x));
        uint32_t yy = expandBits(static_cast<uint32_t>(y));
        uint32_t zz = expandBits(static_cast<uint32_t>(z));

        return (xx << 2) | (yy << 1) | zz;
    }

    void reorderByMorton(std::span<TriangleData> tris)
    {
        Bounds3f bounds = bbEmpty();
        for (auto& t : tris)
            bounds = bbUnion(bounds, Bounds3f{min(min(t.v0, t.v1), t.v2), max(max(t.v0, t.v1), t.v2)});

        auto const getMortonIndex = [bounds](TriangleData const& t) -> uint32_t {
            Point3f  c = (t.v0 + t.v1 + t.v2) / 3;
            Point3f  n = bounds.offset(c); // Normalize to [0,1]
            uint32_t m = morton3D(n.x, n.y, n.z);
            return m;
        };

        std::sort(tris.begin(), tris.end(), [&getMortonIndex](auto const& a, auto const& b) {
            return getMortonIndex(a) < getMortonIndex(b);
        });
    }

    ScanlineRange2D::Iterator::Iterator() : m_p({{-1, -1}}), m_res({{-2, -2}}) {}

    ScanlineRange2D::Iterator::Iterator(Point2i p, Point2i res) : m_p(p), m_res(res) {}

    ScanlineRange2D::Iterator::value_type ScanlineRange2D::Iterator::operator*() const { return m_p; }

    ScanlineRange2D::Iterator& ScanlineRange2D::Iterator::operator++()
    {
        ++m_p.x;
        if (m_p.x >= m_res.x)
        {
            m_p.x = 0;
            ++m_p.y;
            if (m_p.y >= m_res.y)
                m_p.x = -1;
        }

        return *this;
    }

    ScanlineRange2D::Iterator ScanlineRange2D::Iterator::operator++(int)
    {
        Iterator temp = *this;
        ++*this;
        return temp;
    }

    bool ScanlineRange2D::Iterator::operator==(End) const { return m_p.x == -1; }
    bool ScanlineRange2D::Iterator::operator==(Iterator const& other) const
    {
        return m_p == other.m_p && m_res == other.m_res;
    }

    ScanlineRange2D::ScanlineRange2D(Point2i resolution) : m_resolution(resolution)
    {
        assert(m_resolution.x > 0 && m_resolution.y > 0 && "Invalid Resolution");
    }

    ScanlineRange2D::Iterator ScanlineRange2D::begin() const { return Iterator({{0, 0}}, m_resolution); }

    ScanlineRange2D::End ScanlineRange2D::end() const { return {}; }

    /// TODO test
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
        while (remaining > 8)
        { // store abs
            _mm256_storeu_ps(fDest, _mm256_andnot_ps(nzero8, _mm256_loadu_ps(fPtr)));
            fPtr += 8;
            fDest += 8;
            remaining -= 8;
        }

        while (remaining > 4)
        {
            _mm_storeu_ps(fDest, _mm_andnot_ps(nzero4, _mm_loadu_ps(fPtr)));
            fPtr += 4;
            fDest += 4;
            remaining -= 4;
        }

        while (remaining != 0)
            *fDest++ = *fPtr++;

        // Second Step: CDF Computation https://en.algorithmica.org/hpc/algorithms/prefix/
        // split the array into blocks of 8 (remaining later)
        uint32_t const numBlocksTimes8 = m_funcCount & ~0x7u;
        fPtr                           = func.data();
        float* cdf                     = m_buffer.get() + m_funcCount;
        float  carry                   = 0.f;

        for (uint32_t i = 0; i < numBlocksTimes8; ++i)
        {
            __m256 x = _mm256_loadu_ps(fPtr);

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
        m_integral        = CDF().back();
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
            for (uint32_t i = 0; i < numBlocksTimes8; ++i)
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
            return std::numeric_limits<double>::quiet_NaN();
        float const   c      = (x - m_min) / (m_max - m_min) * m_funcCount;
        int32_t const offset = dmt::fl::clamp(static_cast<int32_t>(c), 0, static_cast<int32_t>(m_funcCount - 1));

        float const delta = c - offset;
        auto        cdf   = CDF();
        return dmt::fl::lerp(delta, cdf[offset], cdf[offset + 1]);
    }

    float PiecewiseConstant1D::sample(float u, float* pdf, int32_t* offset) const
    {
        auto cdf  = CDF();
        auto func = absFunc();

        int32_t const off = -1; // TODO SIMD part
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
} // namespace dmt

namespace dmt::test {
    void bvhTestRays(dmt::BVHBuildNode* rootNode)
    {
        Context ctx;
        assert(ctx.isValid() && "Invalid context");
        std::vector<Ray> testRays = {
            // Straight through center of scene
            Ray({{0.5f, 0.5f, -1.0f}}, {{0, 0, 1}}),
            Ray({{1.5f, 1.5f, -1.0f}}, {{0, 0, 1}}),
            Ray({{2.5f, 2.5f, -1.0f}}, {{0, 0, 1}}),
            Ray({{3.5f, 3.5f, -1.0f}}, {{0, 0, 1}}),

            // Grazing edges
            Ray({{1.0f, 1.0f, -1.0f}}, {{0, 0, 1}}),
            Ray({{4.0f, 4.0f, -1.0f}}, {{0, 0, 1}}),

            // Missing all
            Ray({{5.0f, 5.0f, -1.0f}}, {{0, 0, 1}}),
            Ray({{-1.0f, -1.0f, -1.0f}}, {{0, 0, 1}}),

            // Diagonal through stack
            Ray({{-1.0f, -1.0f, 1.0f}}, {{1, 1, 1}}),
            Ray({{0.5f, 0.5f, 0.5f}}, {{1, 1, 1}}),

            // Through nested box
            Ray({{1.0f, 1.0f, 1.0f}}, {{0, 1, 0}}),
        };

        for (size_t i = 0; i < testRays.size(); ++i)
        {
            BVHBuildNode* hit = bvh::traverseBVHBuild(testRays[i], rootNode);
            if (hit)
            {
                ctx.log("Ray {} hit leaf bounding box: min = ({}, {}, {}), max = ({}, {}, {})",
                        std::make_tuple(i,
                                        hit->bounds.pMin.x,
                                        hit->bounds.pMin.y,
                                        hit->bounds.pMin.z,
                                        hit->bounds.pMax.x,
                                        hit->bounds.pMax.y,
                                        hit->bounds.pMax.z));
            }
            else
            {
                ctx.log("Ray {} missed the scene.", std::make_tuple(i));
            }
        }
    }
} // namespace dmt::test

namespace dmt::bvh {
    BVHBuildNode* traverseBVHBuild(Ray ray, BVHBuildNode* bvh, std::pmr::memory_resource* memory)
    {
        std::pmr::vector<BVHBuildNode*> activeNodeStack;
        activeNodeStack.reserve(64);
        activeNodeStack.push_back(bvh);

        BVHBuildNode* intersection = nullptr;
        while (!activeNodeStack.empty())
        {
            BVHBuildNode* current = activeNodeStack.back();
            activeNodeStack.pop_back();

            if (current->childCount > 0)
            {
                // children order of traversal: 1) Distance Heuristic: from smallest to highest tmin - ray origin 2) Sign Heuristic
                // start with distance heuristic
                struct
                {
                    uint32_t i = static_cast<uint32_t>(-1);
                    float    d = fl::infinity();
                } tmins[BranchingFactor];
                uint32_t currentIndex = 0;

                for (uint32_t i = 0; i < current->childCount; ++i)
                {
                    float tmin = fl::infinity();
                    if (slabTest(ray.o, ray.d, current->children[i]->bounds, &tmin))
                    {
                        tmins[currentIndex].d = tmin;
                        tmins[currentIndex].i = i;
                        ++currentIndex;
                    }
                }

                std::sort(std::begin(tmins), std::begin(tmins) + currentIndex, [](auto const& a, auto const& b) {
                    return a.d > b.d;
                });

                for (uint32_t i = 0; i < currentIndex; ++i)
                    activeNodeStack.push_back(current->children[tmins[i].i]);
            }
            else
            {
                // TODO handle any-hit, closest-hit, ...
                // for now, stop at the first leaf intersection
                intersection = current;
                break;
            }
        }

        return intersection;
    }
} // namespace dmt::bvh

namespace dmt::numbers {
    static __m128i permuteSSE2(__m128i i, uint32_t l, uint64_t p)
    {
        uint64_t const p0 = static_cast<uint32_t>(p);
        uint64_t const p1 = p >> 32;

        uint64_t w = l - 1;
        w |= w >> 1;
        w |= w >> 2;
        w |= w >> 4;
        w |= w >> 8;
        w |= w >> 16;
        w |= w >> 32;

        __m128i const W = _mm_set1_epi64x(w);
        __m128i       x = i;

        __m128i const P0 = _mm_set1_epi64x(p0);
        __m128i const P1 = _mm_set1_epi64x(p1);
        __m128i const P  = _mm_set1_epi64x(p);

        __m128i const M1  = _mm_set1_epi64x(0xe170893d);
        __m128i const M2  = _mm_set1_epi64x(0x0929eb3f);
        __m128i const M3  = _mm_set1_epi64x(0x6935fa69);
        __m128i const M4  = _mm_set1_epi64x(0x74dcb303);
        __m128i const M5  = _mm_set1_epi64x(0x9e501cc3);
        __m128i const M6  = _mm_set1_epi64x(0xc860a3df);
        __m128i const ONE = _mm_set1_epi64x(1);

        do
        {
            x = _mm_xor_si128(x, P0);
            x = _mm_mul_epu32(x, M1);
            x = _mm_xor_si128(x, P1);
            x = _mm_xor_si128(x, _mm_srli_epi64(_mm_and_si128(x, W), 4));
            x = _mm_xor_si128(x, _mm_srli_epi64(P0, 8));
            x = _mm_mul_epu32(x, M2);
            x = _mm_xor_si128(x, _mm_srli_epi64(P, 23));
            x = _mm_xor_si128(x, _mm_srli_epi64(_mm_and_si128(x, W), 1));
            x = _mm_mul_epu32(x, _mm_or_si128(ONE, _mm_srli_epi64(P, 27)));
            x = _mm_mul_epu32(x, M3);
            x = _mm_xor_si128(x, _mm_srli_epi64(_mm_and_si128(x, W), 11));
            x = _mm_mul_epu32(x, M4);
            x = _mm_xor_si128(x, _mm_srli_epi64(_mm_and_si128(x, W), 2));
            x = _mm_mul_epu32(x, M5);
            x = _mm_xor_si128(x, _mm_srli_epi64(_mm_and_si128(x, W), 2));
            x = _mm_mul_epu32(x, M6);
            x = _mm_and_si128(x, W);
            x = _mm_xor_si128(x, _mm_srli_epi64(x, 5));
        } while ((_mm_movemask_epi8(_mm_cmpgt_epi64(x, _mm_set1_epi64x(l - 1))) & 0xFFFF) != 0);

        // Return (x + p) % l
        x = _mm_add_epi64(x, P);
        x = _mm_rem_epu64(x, _mm_set1_epi64x(l)); // emulate % l

        return x;
    }

    uint16_t permutationElement(int32_t i, int32_t j, uint32_t l, uint64_t p)
    {
        __m128i v   = _mm_set_epi64x(static_cast<uint64_t>(j), static_cast<uint64_t>(i));
        __m128i res = permuteSSE2(v, l, p);

        alignas(16) uint64_t out[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(out), res);
        return static_cast<uint16_t>(out[0]); // or return both if needed
    }
} // namespace dmt::numbers