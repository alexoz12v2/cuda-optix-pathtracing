#pragma once

#include "cudautils/cudautils-macro.h"
#include "cudautils/cudautils-vecmath.h"

// u = random uniformly distributed fp32 number between 0 and 1
namespace dmt {
    DMT_CORE_API DMT_CPU_GPU DMT_FORCEINLINE inline Point2f cartesianFromPolar(float rho, float phi)
    {
        return {rho * cosf(phi), rho * sinf(phi)};
    }

    // using elevation angle as theta
    DMT_CORE_API DMT_CPU_GPU DMT_FORCEINLINE inline Vector3f cartesianFromSpherical(float rho, float phi, float theta)
    {
        float sin_phi   = sinf(phi);
        float cos_phi   = cosf(phi);
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        return Vector3f(rho * sin_phi * cos_theta, rho * sin_phi * sin_theta, rho * cos_phi);
    }

    DMT_CORE_API DMT_CPU_GPU DMT_FORCEINLINE inline float sin_sqr_to_one_minus_cos(float const s_sq)
    {
        // Using second-order Taylor expansion at small angles for better accuracy.
        return s_sq > 0.0004f ? 1.0f - fl::safeSqrt(1.0f - s_sq) : 0.5f * s_sq;
    }

    DMT_CORE_API DMT_CPU_GPU float sphereLightPDF(float distSqr, float radiusSqr, Vector3f n, Vector3f rayD, bool hadTransmission);
    DMT_CORE_API DMT_CPU_GPU Point2f mapToSphere(Normal3f co);
    DMT_CORE_API DMT_CPU_GPU bool    raySphereIntersect(
           Point3f  rayO,
           Vector3f rayD,
           float    tMin,
           float    tMax,
           Point3f  sphereC,
           float    sphereRadius,
           Point3f* isect_p,
           float*   isect_t);

    /// FROM CYCLES
    /// Uniformly sample a direction in a cone of given angle around `N`. Use concentric mapping to
    /// better preserve stratification. Return the angle between `N` and the sampled direction as
    /// `cos_theta`.
    /// Pass `1 - cos(angle)` as argument instead of `angle` to alleviate precision issues at small
    /// angles (see sphere light for reference).
    DMT_CORE_API DMT_CPU_GPU Vector3f
        sampleUniformCone(Vector3f const N, float const one_minus_cos_angle, Point2f rand, float* cos_theta, float* pdf);

    DMT_CORE_API DMT_CPU_GPU Vector3f sampleUniformSphere(Point2f const rand);

    DMT_CORE_API DMT_CPU_GPU Point2f  sampleUniformDisk(Point2f u);
    DMT_CORE_API DMT_CPU_GPU Vector3f sampleCosHemisphere(Vector3f n, Point2f u, float* pdf = nullptr);
    DMT_CORE_API DMT_CPU_GPU float    cosHemispherePDF(Vector3f n, Vector3f d);

    DMT_CORE_API DMT_CPU_GPU Vector3f sampleUniformHemisphere(Vector3f n, Point2f u, float* pdf = nullptr);

    DMT_CORE_API DMT_CPU_GPU inline Point3f hemisphereFromDisk(Point2f p)
    {
        return {p.x, p.y, fl::safeSqrt(1.f - dotSelf(p))};
    }
} // namespace dmt

namespace dmt::sampling {
    /**
     * normalized (PDF) approximation of the <a href="https://en.wikipedia.org/wiki/Luminous_efficiency_function">Photopic Luminous Efficiency Function</a>
     * graph link: https://www.desmos.com/calculator/fxlr1ce6sj
     * @param sampled wavelength
     * @returns PDF value at the given wavelength
     */
    DMT_CORE_API DMT_CPU_GPU float    visibleWavelengthsPDF(float lambda);
    DMT_CORE_API DMT_CPU_GPU Vector4f visibleWavelengthsPDF(Vector4f lambda);

    /**
     * Sample a wavelength value by means of the <a href="https://en.wikipedia.org/wiki/Inverse_transform_sampling">Inverse Transform Method</a>
     * according to the distribution defined by `visibleWavelengthsPDF`, which approximates the <a href="https://en.wikipedia.org/wiki/Luminous_efficiency_function">Photopic Luminous Efficiency Function</a>
     * of the human eye
     * function Link: https://www.desmos.com/calculator/bufctwl1mv
     * @param u uniformly distributed float in the [0, 1] range
     * @returns sampled wavelength according to the approximated V lambda PDF
     */
    DMT_CORE_API DMT_CPU_GPU float    sampleVisibleWavelengths(float u);
    DMT_CORE_API DMT_CPU_GPU Vector4f sampleVisibleWavelengths(Vector4f u);
} // namespace dmt::sampling
