#pragma once

#include "core/core-macros.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-light.h"

/// this file applies paper <https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf>

namespace dmt {
    // TODO move the *non* build version to cudautils
    /// Build Layout for a class representing the spatial influence of an aggregate of lights through
    /// - spatial bounds (AABB)
    /// - angular bounds (normal directions cone, emission falloff angle)
    struct LightBounds
    {
        Bounds3f bounds;     /// spatial bounds of the light cluster
        Vector3f w;          /// central normal direction of the normal cone of the light cluster
        float    cosTheta_o; /// cosine of normal cone angle of the light cluster
        float    cosTheta_e; /// cosine of emission falloff angle of the light cluster
        float    phi;        /// emitted radiant flux from the light cluster (total light cluster energy)
        bool     twoSided;   /// if true, also cone opposite angles should be considered
    };

    // -- `LightBounds` Methods --
    DMT_CORE_API LightBounds        lbUnion(LightBounds const& lb0, LightBounds const& lb1);
    DMT_CORE_API DMT_FASTCALL float lbImportance(LightBounds const& lb0, Point3f p, Normal3f n);

    // -- `LightBounds` Factory Methods, for each finite position light type --
    DMT_CORE_API LightBounds makeLBFromLight(Light const& light);

    /// Build Layout of a Binary Light Tree Build Node
    struct LightTreeBuildNode
    {
        LightBounds lb;
        float       varPhi;
        union Content
        {
            LightTreeBuildNode* children[2];
            struct LeafData
            {
                Light const* ptr;
                uint32_t     idx;
            } emitter;
        } data;
        uint32_t numEmitters : 31;
        uint32_t leaf        : 1;
    };

    // -- LightTreeBuildNode utils --
    DMT_CORE_API DMT_FASTCALL float adaptiveSplittingHeuristic(LightTreeBuildNode const& node, Point3f p);

    /// Paper, formula section 4.2 - 4.4
    DMT_CORE_API DMT_FASTCALL float summedAreaOrientationHeuristic(
        LightBounds const& lbLeft,
        LightBounds const& lbRight,
        float              parent_Kr,
        float              parent_Ma,
        float              parent_Momega);

    inline constexpr uint32_t MaxNumLightsInCut     = 4;
    inline constexpr uint32_t LightTreeNumBins      = 32;
    inline constexpr uint32_t LightTreeMaxSplitSize = 4; // equal to the number of shadow rays to trace!

    /// returns the index to root
    struct LightTrailPair
    {
        uint32_t lightIdx;
        uint32_t trail;
    };

    /// note: `LightTrailPair` array is sorted on lightIdx, so that you can use `std::binary_search` when computing PMF from bit trail
    DMT_CORE_API size_t lightTreeBuild(std::span<Light>                      lights,
                                       std::pmr::vector<LightTreeBuildNode>* nodes,
                                       std::pmr::vector<LightTrailPair>*     bitTrails,
                                       std::pmr::memory_resource*            temp = std::pmr::get_default_resource());

    struct LightSplit
    {
        LightTreeBuildNode const* nodes[LightTreeMaxSplitSize];
        uint32_t                  count;
    };

    /// Takes precision (0 - 1), shading point (p) and LightTree Root to apply Adaptive Splitting Heuristic to decide how many lights we want
    /// to sample from the Light Tree
    DMT_CORE_API LightSplit lightTreeAdaptiveSplit(LightTreeBuildNode const& ltRoot,
                                                   Point3f                   p,
                                                   float                     precision = 0.5f,
                                                   std::pmr::memory_resource* memory = std::pmr::get_default_resource());

    struct SelectedLights
    {
        uint32_t indices[LightTreeMaxSplitSize];
        float    pmfs[LightTreeMaxSplitSize];
        uint32_t count;
    };

    /// input variables here take inspiration from PBRT-v4's <Declare common variables for light BVH traversal> (page 804)
    /// returns the index of the sampled lights if any, with their PMFs
    DMT_CORE_API SelectedLights DMT_FASTCALL
        selectLightsFromSplit(LightSplit const& lightSplit, Point3f p, Vector3f n, float u, float startPMF);

    /// Compute PMF of a light from its index (position in the span passed in the `build` method) using its light trail
    /// this assumes you didn't select an infinite light, therefore you pass a startPMF = 1 - infiniteLightPMF
    DMT_CORE_API float lightSelectionPMF(LightTreeBuildNode const& ltRoot, Point3f p, Vector3f n, uint32_t trail, float startPMF);
} // namespace dmt