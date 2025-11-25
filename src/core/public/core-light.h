#pragma once

#include "core/core-macros.h"
#include "core/core-math.h"
#include "core/cudautils/cudautils-transform.cuh"

// Note: InfiniteImageLight are treated separately from lights in scene

namespace dmt {
    // -- Constants --

    inline constexpr float threshold = 1e-4f;

    // -- Types --

    struct PointLight
    {
        float radius;  /// we treat the point light as a small sphere, hence it has a radius
        float evalFac; /// multiplicative factor for light strength
    };
    static_assert(std::is_trivial_v<PointLight> && std::is_standard_layout_v<PointLight>);

    // attenuation follows the third grade smoothstep function
    struct SpotLight
    {
        Vector3f direction;           /// cone direction (unit vector) (if storage saving needed, use octahedral)
        float    radius;              /// apex is treated as a small sphere of this radius
        float    evalFac;             /// multiplicative factor for light strength
        float    cosHalfSpotAngle;    /// cosTheta0 (might need also half cotangent)
        float    cosHalfLargerSpread; /// cosThetaE, penumbra
    };
    static_assert(std::is_trivial_v<SpotLight> && std::is_standard_layout_v<SpotLight>);

    enum class LightType : int32_t
    {
        ePoint = 0,
        eEnv,
        eSpot,
        eDistant,
        eMesh,

        Count
    };

    // probably moved in cudautils if used on GPU without any variations
    struct alignas(16) Light
    {
        alignas(16) Transform lightFromRender; // has position in last column (we assume that this is affine), but we keep it anyways
        Point3f   co;
        RGB       strength;
        LightType type;
        union DMT_CORE_API LightData
        {
            PointLight point;
            SpotLight  spot;
        } data;
    };

    // infinite light: image resolution ratio 2:1, POT.
    // given sphercal coordinates of ray direction theta (elevation angle) and phi (azimuthal angle)
    // u = 0.5 * (1 - phi / pi), v = 1 - theta / pi
    // optionally, you also need to know the scene radius and center, if you want to offset somethings
    // - each direction is equally probable, hence PDF = 1 / 4pi
    // - sampling:
    //    1) we want to prefer direction in which there's lots of radiance in the image, therfore we'll construct a 2D PDF from the given buffer
    //    2) sample from that distribution, and return the PDF associated to the position
    //    3) divide PDF by 4pi
    //    4) compute emitted radiance in the sampled point (meaning sample image)
    // This light will still have a lightFromRender, because I can rotate the sphere
    // TODO: start with nearest filtering, then swap for Elliptic Weighted Average (wrap mode == octahedral sphere)
    struct EnvLight
    {
        DMT_CORE_API EnvLight(RGB const*                 image,
                              int32_t                    xRes,
                              int32_t                    yRes,
                              Quaternion                 quat,
                              float                      scale,
                              std::pmr::memory_resource* memory = std::pmr::get_default_resource(),
                              std::pmr::memory_resource* temp   = std::pmr::get_default_resource());

        PiecewiseConstant2D distrib;         /// radiance distrib on image
        Quaternion          lightFromRender; /// assuming infinite radius, we care about orientation
        RGB const*          imageBuffer;
        int32_t             xResolution;
        int32_t             yResolution;
        float               evalFac;
    };

    /// packs necessary information to execute a sampling algorithm for a chosen light
    struct LightSampleContext
    {
        Ray      ray;             /// last ray from main path
        Point3f  p;               /// position of last surface intersection
        Vector3f n;               /// surface normal (geometric?)
        bool     hadTransmission; /// flag which tells us if the given path had at least a transmission
    };

    /// Incident radiance (RGB) is evaluated separately starting from this sample class
    struct LightSample
    {
        Point3f   p;   /// position for light, direction if directional/distant light (which we don't currently have)
        Vector3f  ng;  /// normal on light
        float     t;   /// distance from light (if not directional)
        Vector3f  d;   /// direction from shading point to light (wi)
        Point2f   uv;  /// parametric coordinates on a primitive (textures on area light? currently not used)
        float     pdf; /// PDF for (selecting light MAYBE) and a point on the light
        float     pdfSelection; /// PDF for selecting light (set BEFORE light sampling, in light selection)
        float     evalFac;      /// Intensity multiplier
        LightType type;         /// type of light;
        bool      delta;
    };

    // note on PDF: first light sample fills its initial value, than the light tree pdf multiplies it, then multiply it by
    //  - float power_heuristic(const float a, const float b) { return (a * a) / (a * a + b * b); }
    //     - mis_weight = shadow_linking_light_sample_mis_weight(kg, state, path_flag, &ls, ray.P);
    //     - bsdf_spectrum = light_eval * mis_weight * INTEGRATOR_STATE(state, shadow_link, dedicated_light_weight);
    // where light eval is the strength

} // namespace dmt

namespace dmt {
    // -- Factory Methods --
    DMT_CORE_API Light makePointLight(Transform const& t, RGB emission, float radius = 1e-3f, float factor = 1 / fl::pi());
    DMT_CORE_API Light makeSpotLight(Transform const& t,
                                     RGB              emission,
                                     float            cosTheta0,
                                     float            cosThetae,
                                     float            radius = 1e-3f,
                                     float            factor = 1 / fl::pi());

    // -- Methods Light Specific --
    // ---- Point Light ----
    /// computes also factor in `evalFac` to multiply to `Light::strength` such that we obtain the desired falloff effect
    /// `sample` used as in-out parameter, as we require it to possess the sampled point `p`
    /// ht -> path had transmission at least once
    DMT_CORE_API bool pointLightSampleFromContext(Light const& light, Point2f u, Point3f p, Vector3f n, bool ht, LightSample* sample);

    /// basically computes ray-sphere intersection. Should be useful when choosing the light to sample from
    DMT_CORE_API bool pointLightIntersect(Light const& light, Ray const& ray, float* t);

    // ---- Spot Light ----
    DMT_CORE_API bool spotLightSampleFromContext(Light const& light, Point2f u, Point3f p, Vector3f n, bool ht, LightSample* sample);

    DMT_CORE_API bool spotLightIntersect(Light const& light, Ray const& ray, float* t);

    // -- Methods --
    /// sample a direct illumination path and its weights given a chose light
    /// takes sampling context, 2x uniformly distributed numbers,
    /// and splits a light sample and whether sampling was successful or not
    DMT_CORE_API bool lightSampleFromContext(Light const& light, LightSampleContext const& lsCtx, Point2f u, LightSample* sample);

    /// use `sample` as an in out parameter to evaluate spectrum for light. what needs to be initialized
    /// - n, evalFac
    DMT_CORE_API RGB lightEval(Light const& light, LightSample const* sample);

    DMT_CORE_API bool lightIntersect(Light const& light, Ray const& ray, float* t);

    // ---- Env Light ----
    /// sample context is useless, kept just to have the sameish api as the other light types
    DMT_CORE_API bool envLightSampleFromContext(EnvLight const&           light,
                                                LightSampleContext const& lsCtx,
                                                Point2f                   u,
                                                LightSample*              sample);
    /// uses uv from light sample to sample the texture in light. The sample should be generated by the same light!
    DMT_CORE_API RGB envLightEval(EnvLight const& light, LightSample const* sample);

    /// start from world space direction (eg ray escaped the scene) and grab sampled radiance and its PDF
    DMT_CORE_API RGB envLightEval(EnvLight const& light, Vector3f wi, float* pdf);
} // namespace dmt
