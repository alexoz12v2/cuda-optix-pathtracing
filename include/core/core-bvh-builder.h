#pragma once

#include "core/core-macros.h" // must be before cudautils

#include "cudautils/cudautils-vecmath.h"

#include "core/core-trianglemesh.h"
#include "core/core-primitive.h"

#include "platform/platform-memory.h"

#include <span>
#include <vector>

namespace dmt {
    inline constexpr int     BranchingFactor       = 8;
    inline constexpr int     SIMDWidth             = 8;
    inline constexpr int     LeavesBranchingFactor = 7;
    inline constexpr int32_t MinNumBin             = 16;
    inline constexpr int32_t MaxNumBin             = 128;
    inline constexpr float   BinScaleFactor        = 2.f;
    static constexpr float   DegenerateEpsilon     = 1e-5f;

    static_assert(SIMDWidth == BranchingFactor, "BVH vectorization doesn't work");

    /// Note: Children are Owned by ancestor node, while primitives are *NOT* owned by the node structure
    struct DMT_CORE_API BVHBuildNode
    {
        Bounds3f         bounds;
        Primitive const* primitives[LeavesBranchingFactor];
        BVHBuildNode*    children[BranchingFactor];
        uint32_t         childCount;
        uint32_t         primitiveCount;
    };

    struct DMT_CORE_API alignas(32) BVHWiVeCluster
    {
        //WiveClusters 8nodes
        //bounding box information
        float bxmin[SIMDWidth];
        float bxmax[SIMDWidth];
        float bymin[SIMDWidth];
        float bymax[SIMDWidth];
        float bzmin[SIMDWidth];
        float bzmax[SIMDWidth];
        //data information
        //32bit*8node->32bit:1bit flag active node,24bit for the permutation info, 1bit flag->inner node/leaf node,6bit for the offset
        uint64_t slotEntries[SIMDWidth]; // sn is split in 2 _m256

        unsigned char _padding[64];
    };
    static_assert(sizeof(BVHWiVeCluster) == 320);
    static_assert(std::is_trivial_v<BVHBuildNode> && std::is_standard_layout_v<BVHBuildNode>,
                  "needed to use aggregate init and memset/memcpy");

    struct DMT_CORE_API alignas(BVHWiVeCluster) BVHWiVeLeaf
    {
        static constexpr uint32_t numTriangles = LeavesBranchingFactor;

        size_t triIdx[numTriangles];

        Point3f v0s[numTriangles];
        Point3f v1s[numTriangles];
        Point3f v2s[numTriangles];

        uint32_t instanceIdx;
        uint8_t  triCount; // 1..8

        //unsigned char _padding[sizeof(BVHWiVeCluster) - 192];
    };
    static_assert(sizeof(BVHWiVeLeaf) == sizeof(BVHWiVeCluster));

#if 0
    struct DMT_CORE_API BVHWiVe
    {
        BVHWiVeCluster bvhClusters[256];
    };
#endif
} // namespace dmt

namespace dmt::bvh {
    /// @warning allocate in _temp_ only stuff which doesn't need destruction!
    DMT_CORE_API BVHBuildNode* build(std::span<Primitive const*> const prims,
                                     std::pmr::memory_resource*        temp,
                                     std::pmr::memory_resource*        memory = std::pmr::get_default_resource());

    DMT_CORE_API void cleanup(BVHBuildNode* node, std::pmr::memory_resource* memory = std::pmr::get_default_resource());

    DMT_CORE_API std::size_t groupTrianglesInBVHLeaves(
        BVHBuildNode*                                node,
        std::pmr::vector<dmt::UniqueRef<Primitive>>& out,
        std::pmr::memory_resource*                   temp,
        std::pmr::memory_resource*                   memory = std::pmr::get_default_resource());

    /// assumes you give it an instance belonging to the scene itself
    /// @warning allocate in _temp_ only stuff which doesn't need destruction!
    DMT_CORE_API BVHBuildNode* buildForInstance(
        Scene const&                            scene,
        size_t                                  instanceIdx,
        std::pmr::vector<UniqueRef<Primitive>>& outPrims,
        std::pmr::memory_resource*              temp,
        std::pmr::memory_resource*              memory = std::pmr::get_default_resource());

    DMT_CORE_API BVHWiVeCluster* buildBVHWive(BVHBuildNode*              bvh,
                                              uint32_t*                  nodeCount,
                                              std::pmr::memory_resource* temp,
                                              std::pmr::memory_resource* memory = std::pmr::get_default_resource());

    DMT_CORE_API BVHBuildNode* buildCombined(BVHBuildNode*              root,
                                             std::span<BVHBuildNode*>   nodes,
                                             std::pmr::memory_resource* temp,
                                             std::pmr::memory_resource* memory = std::pmr::get_default_resource());
    //DMT_CORE_API void          traverseRay(Ray ray, BVHWiVe* bvh, std::pmr::memory_resource* _temp);

    DMT_CORE_API bool traverseRay(Ray const&                 ray,
                                  BVHWiVeCluster const*      bvh,
                                  uint32_t                   nodeCount,
                                  uint32_t*                  instanceIdx,
                                  size_t*                    triIdx,
                                  triangle::Triisect*        outTri,
                                  std::pmr::memory_resource* temp = std::pmr::get_default_resource());
} // namespace dmt::bvh