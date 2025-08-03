#pragma once

#include "core/core-macros.h"
#include "core/core-cudautils-cpubuild.h"

/**
 * Important Note: Each relative IOR specifies insideIOR / outsideIOR, where the "outside" is defined by the normal direction
 */
namespace dmt {

    inline Vector3f safeNormalizeFallback(Vector3f const a, Vector3f const fallback)
    {
        float const t = normL2(a);
        return (t != 0.0f) ? a * (1.0f / t) : fallback;
    }

    inline Vector3f reflect(Vector3f wo, Vector3f n) { return -wo + 2 * dot(wo, n) * n; }

    /// eta = Relative IOR of the material's interior.
    /// etap = out parameter equal to eta if outside -> inside, or equal to 1/eta if inside -> outside
    DMT_CORE_API bool refract(Vector3f wi, Normal3f n, float eta, float* etap, Vector3f* wt);

    /// Taken by cycles
    /// If the shading normal results in specular reflection in the lower hemisphere, raise the shading
    /// normal towards the geometry normal so that the specular reflection is just above the surface.
    /// Only used for glossy materials.
    DMT_CORE_API Vector3f ensureValidSpecularReflection(Vector3f const Ng, Vector3f const I, Vector3f N);

    struct BsdfClosure
    {
        RGB      weight;       // equal to reflectance/transmittance
        Vector3f N;            // (microfacet if GGX)normal used for formulas
        float    sampleWeight; // typically multiplied by 1/PDF of sampled incident direction
    };

    DMT_CORE_API BsdfClosure makeClosure(RGB weight);
} // namespace dmt

// TODO test these function

/// a more involved implementation is found in cycles, method `microfacet_fresnel` in `bsdf_microfacet.h`
namespace dmt::fresnel {
    /// https://en.wikipedia.org/wiki/Fresnel_equations#Power_(intensity)_reflection_and_transmission_coefficients
    /// @note: cosine is clamped -1,1 and, if negative, flips the relative IOR before computing the fresnel reflectance
    DMT_CORE_API float reflectanceDielectric(float cosThetai, float relIOR);

    /// @note: cosine is clamped 0,1 and here cannot be negative, as we assume conductors to be fully opaque
    /// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    /// $\eta=\frac{n_t}{n_i}$, $\eta_k=\frac{k_t}{n_i}$
    DMT_CORE_API RGB DMT_FASTCALL reflectanceConductor(float cosThetai, RGB eta, RGB etak);
} // namespace dmt::fresnel

// Oren Nayar assumes diffuse, fully opacity (no transmission)
namespace dmt::oren_nayar {
    struct BRDF
    {
        BsdfClosure closure;

        // Note: These are personal fixes, cause somehow the formula doesn't work
        RGB   albedo;
        float multiscatterMultiplier;

        // official stuff
        RGB   multiscatterTerm;
        float roughness;
        float a;
        float b;
    };

    /// Evaluate, from qualitative parameters, the parametric representation of an Oren-Nayar BRDF surface
    /// roughness -> \sigma in paper (note: If 0 is given, it's equivalent to Lambert BRDF with additional cost)
    /// color     -> \rho albedo
    DMT_CORE_API BRDF makeParams(float    roughness,
                                 RGB      color,
                                 Vector3f ns,
                                 Vector3f wi,
                                 float    multiscatterMultiplier = 1.f,
                                 RGB      weight                 = RGB::fromVec(Vector3f::one()));

    /// ng geometric normal
    /// u  point uniformly distributed in unit square [0,1]
    /// ns shading normal
    /// sample incident direction wi using cosine weighted hemisphere sampling
    DMT_CORE_API Vector3f DMT_FASTCALL sample(Vector3f ns, Vector3f ng, Point2f u, float* pdf = nullptr);


    // NOTE: This implements the improved Oren-Nayar model by Yasuhiro Fujii
    // (https://mimosa-pudica.net/improved-oren-nayar.html), plus an
    // energy-preserving multi-scattering term based on the OpenPBR specification
    // (https://academysoftwarefoundation.github.io/OpenPBR)
    DMT_CORE_API float DMT_FASTCALL G(float cosTheta);

    /// Compute reflectance given outgoing and incident direction using the Oren-Nayar BRDF model
    DMT_CORE_API RGB intensity(BRDF const& params, Vector3f wo, Vector3f wi);
} // namespace dmt::oren_nayar

namespace dmt::ggx {
    struct BSDF
    {
        BsdfClosure closure;
        Vector3f    T;

        float alphax, alphay;

        // used to account for the missing energy due to the single-scattering microfacet model
        float energyScale;
        float eta;

        // handle fresnel
        int32_t isConductor;
        union UFresnelData
        {
            struct Conductor
            {
                RGB eta, etak;
            } c;
            struct Dielectric
            {
                RGB reflectanceTint; // different from (1,1,1) if caustics
                RGB transmittanceTint;
            } d;
        } fresnel;
    };

    struct BSDFSample
    {
        /// sampled incident direction
        Vector3f wi;

        /// sampled (micro) normal direction
        Vector3f wm;

        /// computed BSDF value for sampled direction and normal
        RGB f;

        /// PDF of sampled value
        float pdf;

        /// relative real part of IOR of intersection. equal to 1 if reflection or conductor
        /// > 1 if dielectric and outside -> inside, < 1 if dielectric and inside -> outside
        float eta;
    };

    /// set closure, alphax and alphay (roughness), energy scale and index of refraction
    /// inspired by cycles' `bsdf_microfacet_setup_fresnel_dielectric_tint`
    DMT_CORE_API BSDF DMT_FASTCALL makeDielectric(
        Vector3f wo,
        Vector3f ns,
        Vector3f ng,
        float    ior,
        float    alphax,
        float    alphay,
        RGB      r,
        RGB      t,
        RGB      weight = RGB::fromVec(Vector3f::one()));

    /// tangent can be any unit vector perpendicular to normal
    DMT_CORE_API BSDF DMT_FASTCALL makeConductor(
        Vector3f wo,
        Vector3f ns,
        Vector3f ng,
        float    alphax,
        float    alphay,
        Vector3f tangent,
        RGB      eta,
        RGB      etak,
        RGB      weight = RGB::fromVec(Vector3f::one()));

    /// @warning assumes `w` is in tangent space
    DMT_CORE_API float DMT_FASTCALL auxiliaryLambda(Vector3f w, float alphax, float alphay);

    /// @warning assumes `w` is in tangent space
    DMT_CORE_API float DMT_FASTCALL smithG1(Vector3f w, float alphax, float alphay);

    /// Smith shadowing-masking term, here in the non-separable form.
    /// For details, see:
    /// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs.
    /// Eric Heitz, JCGT Vol. 3, No. 2, 2014.
    /// https://jcgt.org/published/0003/02/03/
    /// @warning assumes `wo` and `wi` is in tangent space
    DMT_CORE_API float DMT_FASTCALL heightCorrG(Vector3f wo, Vector3f wi, float alphax, float alphay);

    /// @warning assumes `wm` is in tangent space
    DMT_CORE_API float DMT_FASTCALL NDF(Vector3f wm, float alphax, float alphay);

    /// @warning assumes `wm` is in tangent space
    DMT_CORE_API float DMT_FASTCALL PDF(Vector3f w, Vector3f wm, float alphax, float alphay);

    /// if this is true, then microsurface is basically planar, hence you shouldn't use roughnesses so low
    DMT_CORE_API inline bool effectivelySmooth(float alphax, float alphay) { return fmaxf(alphax, alphay) < 1e-3f; }

    /// GGX VNDF importance sampling algorithm from: Sampling the GGX Distribution of Visible Normals.
    /// Eric Heitz, JCGT Vol. 7, No. 4, 2018. https://jcgt.org/published/0007/04/01/
    /// cannot return PDF because it depends on
    /// - does it refract or not?
    /// - if it refracts, is it singular or not (see cycles, bsdf_microfacet.h)
    /// It is important to be able to give a PDF Though, so this must be wrapped in a sampling procedure which knows the
    /// nature of the material using this distribution, cause if zero we put zero as reflectance/transmittance
    DMT_CORE_API Vector3f DMT_FASTCALL sampleMicroNormal(Vector3f wi, Point2f u, float alphax, float alphay);

    // TODO: sample and evaluate for dielectric and conductor
    /// eta is an indicator of refraction. it'll be one if refraction happened
    /// u, uc uniformly distributed in [0,1]
    DMT_CORE_API BSDFSample DMT_FASTCALL sample(BSDF const& bsdf, Vector3f w, Vector3f ng, Point2f u, float uc);

    /// @Warning: here wi is the incoming direction of view vector, not incoming direction of light, as used in other functions
    DMT_CORE_API RGB DMT_FASTCALL eval(BSDF const& bsdf, Vector3f wo, Vector3f wi, Vector3f ng, float* pdf);
} // namespace dmt::ggx