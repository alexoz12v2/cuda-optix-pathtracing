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
    inline constexpr int     LeavesBranchingFactor = 12;
    inline constexpr int32_t MinNumBin             = 16;
    inline constexpr int32_t MaxNumBin             = 128;
    inline constexpr float   BinScaleFactor        = 2.f;
    static constexpr float   DegenerateEpsilon     = 1e-5f;

    /// Note: Children are Owned by ancestor node, while primitives are *NOT* owned by the node structure
    struct DMT_CORE_API BVHBuildNode
    {
        Bounds3f         bounds;
        Primitive const* primitives[LeavesBranchingFactor];
        BVHBuildNode*    children[BranchingFactor];
        uint32_t         childCount;
        uint32_t         primitiveCount;
    };
    static_assert(std::is_trivial_v<BVHBuildNode> && std::is_standard_layout_v<BVHBuildNode>,
                  "needed to use aggregate init and memset/memcpy");
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

    DMT_CORE_API BVHBuildNode* buildCombined(BVHBuildNode*              root,
                                             std::span<BVHBuildNode*>   nodes,
                                             std::pmr::memory_resource* temp,
                                             std::pmr::memory_resource* memory = std::pmr::get_default_resource());
} // namespace dmt::bvh