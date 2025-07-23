#include "utilities.h"

#include "platform/platform-context.h"

#if !defined(DMT_ARCH_X86_64)
    #error "what"
#endif

#include <emmintrin.h> // SSE2

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