#pragma once

#include "core/core-macros.h"
#include "cudautils/cudautils-vecmath.h"
#include "core/core-primitive.h"
#include "platform/platform-memory.h"

#include <span>
#include <vector>

namespace dmt {
    inline constexpr int LogBranchingFactor    = 3;
    inline constexpr int BranchingFactor       = 1 << LogBranchingFactor;
    inline constexpr int LeavesBranchingFactor = 1 << LogBranchingFactor; // TODO better
    inline constexpr int TrianglesPerLeaf      = 4;
    inline constexpr int maxPrimsInNode        = 8;

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
    DMT_CORE_API BVHBuildNode* build(std::span<Primitive const*> const prims,
                                     std::pmr::memory_resource* temp, // allocate only stuff which doesn't need destruction!
                                     std::pmr::memory_resource* memory = std::pmr::get_default_resource());

    DMT_CORE_API void cleanup(BVHBuildNode* node, std::pmr::memory_resource* memory = std::pmr::get_default_resource());

    DMT_CORE_API std::size_t groupTrianglesInBVHLeaves(
        BVHBuildNode*                                node,
        std::pmr::vector<dmt::UniqueRef<Primitive>>& out,
        std::pmr::memory_resource*                   temp,
        std::pmr::memory_resource*                   memory = std::pmr::get_default_resource());
} // namespace dmt::bvh