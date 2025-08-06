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
        float    phi;        /// emitted radiant flux from the light cluster
        bool     twoSided;   /// if true, also cone opposite angles should be considered
    };

    // -- `LightBounds` Methods --
    DMT_CORE_API LightBounds        lbUnion(LightBounds const& lb0, LightBounds const& lb1);
    DMT_CORE_API DMT_FASTCALL float lbImportance(LightBounds const& lb0, Point3f p, Normal3f n);

    // -- `LightBounds` Factory Methods, for each finite position light type --
    DMT_CORE_API LightBounds makeLBFromLight(Light const& light);

    // -- `LightBounds` Energy Variance Methods, for each finite position light type --
    /// `PDF_Emission(x,w) = PDF_Pos(x) * PDF_dir(w | x)`
    /// once you know the emission PDF for a given light source, you can analitically (or numerically) compute its
    /// energy variance and store it inside the light tree node for adaptive splitting (lightcuts)
    DMT_CORE_API float energyVarianceFromLight(Light const& light);
} // namespace dmt