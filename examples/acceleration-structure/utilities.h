#pragma once

#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-math.h"
#include "core/core-primitive.h"
#include "core/core-dstd.h"
#include "core/core-bsdf.h"
#include "core/core-light.h"

#include <memory_resource>

namespace dmt {
    // intersection/math algorithms
    bool DMT_FASTCALL slabTest(Point3f         rayOrigin,
                               Vector3f        rayDirection,
                               Bounds3f const& box,
                               float*          outTmin = nullptr,
                               float*          outTmax = nullptr);

    std::pmr::vector<dmt::UniqueRef<Primitive>> makeSinglePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource());

    std::pmr::vector<dmt::UniqueRef<Primitive>> makePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource());

    uint32_t morton3D(float x, float y, float z);

    void reorderByMorton(std::span<TriangleData> tris);

    class ScanlineRange2D
    {
    public:
        struct End
        {
        };
        class Iterator
        {
        public:
            using difference_type   = std::ptrdiff_t;
            using value_type        = Point2i;
            using iterator_category = std::forward_iterator_tag;

            Iterator();
            Iterator(Point2i p, Point2i res);

            value_type operator*() const;

            Iterator& operator++();
            Iterator  operator++(int);

            bool operator==(End) const;
            bool operator==(Iterator const& other) const;

        private:
            Point2i m_p;
            Point2i m_res;
        };
        static_assert(std::forward_iterator<Iterator>, "Failed");

    public:
        explicit ScanlineRange2D(Point2i resolution);

        Iterator begin() const;
        End      end() const;

    private:
        Point2i m_resolution;
    };

    // if buffer nullptr, query resolution
    bool openEXR(os::Path const&            imagePath,
                 RGB**                      buffer,
                 int32_t*                   xRes,
                 int32_t*                   yRes,
                 std::pmr::memory_resource* temp = std::pmr::get_default_resource());

    bool writePNG(os::Path const&            imgPath,
                  RGB const*                 buffer,
                  int32_t                    xRes,
                  int32_t                    yRes,
                  std::pmr::memory_resource* temp = std::pmr::get_default_resource());
} // namespace dmt

namespace dmt::test {
    void testGGXconductor(uint32_t numSamples, Vector3f wo = normalize(Vector3f{-1, -1, 1}));
    void bvhTestRays(BVHBuildNode* rootNode);
    void testDistribution1D();
    void testDistribution2D();
    void testOctahedralProj();
    void testIndexedTriangleGrouping();
    void testSphereLightPDFAnalyticCheck();
    void testEnvironmentalLightConstantValue();
    void testQuaternionRotation();
    void testMipmappedTexturePrinting();
} // namespace dmt::test

namespace dmt::bvh {
    BVHBuildNode*    traverseBVHBuild(Ray                        ray,
                                      BVHBuildNode*              bvh,
                                      std::pmr::memory_resource* memory = std::pmr::get_default_resource());
    Primitive const* intersectBVHBuild(Ray                        ray,
                                       BVHBuildNode*              bvh,
                                       Intersection*              outIsect = nullptr,
                                       std::pmr::memory_resource* memory   = std::pmr::get_default_resource());

    std::pmr::vector<Primitive const*> extractPrimitivesFromBuild(
        BVHBuildNode*              bvh,
        std::pmr::memory_resource* memory = std::pmr::get_default_resource());

    Primitive const* intersectWideBVHBuild(Ray ray, BVHBuildNode* bvh, Intersection* outIsect);

    void traverseRay(Ray ray, BVHWiVeSoA* bvh, std::pmr::memory_resource* _temp);
} // namespace dmt::bvh

namespace dmt::numbers {
    template <std::unsigned_integral... Args>
    constexpr uint64_t DMT_FASTCALL hashIntegers(Args... args)
        requires(sizeof...(Args) > 0)
    {
        uint64_t hash    = 0;
        auto     combine = [](uint64_t seed, uint64_t value) -> uint64_t {
            return seed ^ (value + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2));
        };

        ((hash = combine(hash, static_cast<uint64_t>(args))), ...);
        return hash;
    }

    DMT_FORCEINLINE constexpr uint64_t mixBits(uint64_t v)
    {
        v ^= (v >> 31);
        v *= 0x7fb5d329728ea185;
        v ^= (v >> 27);
        v *= 0x81dadef4bc2dd44d;
        v ^= (v >> 33);
        return v;
    }

    uint16_t permutationElement(int32_t i, int32_t j, uint32_t l, uint64_t p);
    uint16_t permutationElement(int32_t i, uint32_t l, uint64_t p);
} // namespace dmt::numbers
