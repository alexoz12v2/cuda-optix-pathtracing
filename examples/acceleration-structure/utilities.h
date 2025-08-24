#pragma once

#include "core/core-bvh-builder.h"
#include "cudautils/cudautils.h"
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

#if 0
    void traverseRay(Ray ray, BVHWiVe* bvh, std::pmr::memory_resource* _temp);
#endif
} // namespace dmt::bvh
