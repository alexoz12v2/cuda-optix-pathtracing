#pragma once

#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-math.h"
#include "core/core-primitive.h"

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
} // namespace dmt

namespace dmt::test {
    void bvhTestRays(BVHBuildNode* rootNode);
}

namespace dmt::bvh {
    BVHBuildNode* traverseBVHBuild(Ray                        ray,
                                   BVHBuildNode*              bvh,
                                   std::pmr::memory_resource* memory = std::pmr::get_default_resource());
}

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
} // namespace dmt::numbers
