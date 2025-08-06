#include "cudautils-numbers.h"

namespace dmt {
    __host__ __device__ float sphereLightPDF(float distSqr, float radiusSqr, Vector3f n, Vector3f rayD, bool hadTransmission)
    {
        static constexpr float _1Over2Pi = 1 / fl::twoPi();
        if (distSqr > radiusSqr)
            return _1Over2Pi / sin_sqr_to_one_minus_cos(radiusSqr / distSqr);
        else
            return hadTransmission ? _1Over2Pi * 0.5f : cosHemispherePDF(n, rayD);
    }

    __host__ __device__ Point2f mapToSphere(Normal3f co)
    {
        static constexpr float _1Over2Pi = 1 / fl::twoPi();

        float const l = dotSelf(co);
        float       u;
        float       v;
        if (l > 0.0f)
        {
            if (co.x == 0.0f && co.y == 0.0f)
            {
                u = 0.0f; /* Otherwise domain error. */
            }
            else
            {
                u = (0.5f - atan2f(co.x, co.y) * _1Over2Pi);
            }
            v = 1.0f - fl::safeacos(co.z / sqrtf(l)) * _1Over2Pi;
        }
        else
        {
            u = v = 0.0f;
        }

        return {u, v};
    }

    __host__ __device__ bool raySphereIntersect(
        Point3f  rayO,
        Vector3f rayD,
        float    tMin,
        float    tMax,
        Point3f  sphereC,
        float    sphereRadius,
        Point3f* isect_p,
        float*   isect_t)
    {
        // courtesy of cycles
        Vector3f const d_vec       = sphereC - rayO;
        float const    r_sq        = sphereRadius * sphereRadius;
        float const    d_sq        = dot(d_vec, d_vec);
        float const    d_cos_theta = dot(d_vec, rayD);

        if (d_sq > r_sq && d_cos_theta < 0.0f)
        {
            // Ray origin outside sphere and points away from sphere.
            return false;
        }

        float const d_sin_theta_sq = dotSelf(d_vec - d_cos_theta * rayD);

        if (d_sin_theta_sq > r_sq)
        {
            // Closest point on ray outside sphere.
            return false;
        }

        // Law of cosines
        float const t = d_cos_theta - copysignf(sqrtf(r_sq - d_sin_theta_sq), d_sq - r_sq);

        if (t > tMin && t < tMax)
        {
            *isect_t = t;
            *isect_p = rayO + rayD * t;
            return true;
        }

        return false;
    }

    __host__ __device__ Vector3f
        sampleUniformCone(Vector3f const N, float const one_minus_cos_angle, Point2f const rand, float* cos_theta, float* pdf)
    {
        if (one_minus_cos_angle > 0)
        {
            // Remap radius to get a uniform distribution w.r.t. solid angle on the cone.
            // The logic to derive this mapping is as follows:
            //
            // Sampling a cone is comparable to sampling the hemisphere, we just restrict theta. Therefore,
            // the same trick of first sampling the unit disk and the projecting the result up towards the
            // hemisphere by calculating the appropriate z coordinate still works.
            //
            // However, by itself this results in cosine-weighted hemisphere sampling, so we need some kind
            // of remapping. Cosine-weighted hemisphere and uniform cone sampling have the same conditional
            // PDF for phi (both are constant), so we only need to think about theta, which corresponds
            // directly to the radius.
            //
            // To find this mapping, we consider the simplest sampling strategies for cosine-weighted
            // hemispheres and uniform cones. In both, phi is chosen as `2pi * random()`. For the former,
            // `r_disk(rand) = sqrt(rand)`. This is just naive disk sampling, since the projection to the
            // hemisphere doesn't change the radius.
            // For the latter, `r_cone(rand) = sin_from_cos(mix(cos_angle, 1, rand))`.
            //
            // So, to remap, we just invert r_disk `(-> rand(r_disk) = r_disk^2)` and insert it into
            // r_cone: `r_cone(r_disk) = r_cone(rand(r_disk)) = sin_from_cos(mix(cos_angle, 1, r_disk^2))`.
            // In practice, we need to replace `rand` with `1 - rand` to preserve the stratification,
            // but since it's uniform, that's fine
            Point2f     xy = sampleUniformDisk(rand);
            float const r2 = dotSelf(xy);

            /* Equivalent to `mix(cos_angle, 1.0f, 1.0f - r2)`. */
            *cos_theta = 1.0f - r2 * one_minus_cos_angle;

            /* Remap disk radius to cone radius, equivalent to `xy *= sin_theta / sqrt(r2)`. */
            xy *= fl::safeSqrt(one_minus_cos_angle * (2.0f - one_minus_cos_angle * r2));

            *pdf = 1.f / (fl::twoPi() * one_minus_cos_angle);

            Vector3f T{};
            Vector3f B{};
            gramSchmidt(N, &T, &B);
            return xy.x * T + xy.y * B + *cos_theta * N;
        }

        *cos_theta = 1.0f;
        *pdf       = 1.0f;

        return N;
    }

    __host__ __device__ Vector3f sampleUniformSphere(Point2f const rand)
    {
        float const z   = 1.0f - 2.0f * rand.x;
        float const r   = fl::safeSqrt(1.f - z * z); // sin from cos
        float const phi = fl::twoPi() * rand.y;

        // polar to cartesian
        float const xCartesian = r * cosf(phi);
        float const yCartesian = r * sinf(phi);

        return {xCartesian, yCartesian, z};
    }

    __host__ __device__ Point2f sampleUniformDisk(Point2f u)
    {
        // remap x,y to -1,1
        float const a = 2.f * u.x - 1.f;
        float const b = 2.f * u.y - 1.f;

        float phi = 0.f;
        float rho = 0.f;
        if (a == 0.f && b == 0.f)
            return {};

        if (a > b)
        {
            rho = a;
            phi = fl::piOver4() * (b / a);
        }
        else
        {
            static constexpr float _3piOver4 = fl::piOver2() - fl::piOver4();
            rho                              = b;
            phi                              = _3piOver4 * (a / b);
        }

        return cartesianFromPolar(rho, phi);
    }

    __host__ __device__ Vector3f sampleCosHemisphere(Vector3f n, Point2f u, float* pdf)
    {
        assert(fl::abs(normL2(n) - 1.f) < 1e-5f && "Expected unit vector");
        Point2f const rand     = sampleUniformDisk(u);              // sine if n is unit
        float const   cosTheta = fl::safeSqrt(1.f - dotSelf(rand)); // length, now I need frame
        Vector3f      T, B;
        gramSchmidt(n, &T, &B);
        if (pdf)
            *pdf = cosTheta * fl::rcpPi();

        return rand.x * T + rand.y * B + cosTheta * n;
    }

    __host__ __device__ Vector3f sampleUniformHemisphere(Vector3f n, Point2f u, float* pdf)
    {
        assert(fl::abs(normL2(n) - 1.f) < 1e-5f && "Expected unit vector");
        Point2f     xy = sampleUniformDisk(u);
        float const z  = 1.f - dotSelf(xy);

        xy *= fl::safeSqrt(z + 1.f);
        Vector3f T{}, B{};
        gramSchmidt(n, &T, &B);

        Vector3f wo = xy.x * T + xy.y * B + z * n;
        if (pdf)
            *pdf = 1 / fl::twoPi();
        return wo;
    }

    __host__ __device__ float cosHemispherePDF(Vector3f n, Vector3f d)
    {
        assert(fl::abs(normL2(n) - 1.f) < 1e-5f && fl::abs(normL2(d) - 1.f) < 1e-5f);
        float const cosTheta = dot(n, d);
        return cosTheta > 0.f ? cosTheta * fl::rcpPi() : 0.f;
    }
} // namespace dmt

namespace dmt::sampling {
    __host__ __device__ float visibleWavelengthsPDF(float lambda)
    {
        float ret = 0.f;
        if (lambda >= 360.f && lambda < 830.f)
        {
            ret = glm::cosh(0.0072f * (lambda - 538));
            ret *= ret;
            ret = fl::rcp(ret);
            ret *= 0.0039398042f;
        }

        return ret;
    }

    __host__ __device__ float sampleVisibleWavelengths(float u)
    {
        assert(u >= 0.f && u <= 1.f);
        return 538 - 138.888889f * glm::atanh(0.85691062f - 1.82750197f * u);
    }

    __host__ __device__ Vector4f visibleWavelengthsPDF(Vector4f lambda)
    {
        Vector4f ret = Vector4f::zero();

        // Mask for values within the valid range [360, 830)
        glm::bvec4 valid = glm::greaterThanEqual(toGLM(lambda), glmLambdaMin()) &&
                           glm::lessThanEqual(toGLM(lambda), glmLambdaMax());
        if (glm::any(valid)) // Check if any values are in the valid range
        {
            glm::vec4 diff    = toGLM(lambda) - 538.f;
            glm::vec4 coshVal = glm::cosh(0.0072f * diff);
            coshVal           = coshVal * coshVal;       // Square the cosh values
            glm::vec4 pdf     = 0.0039398042f / coshVal; // Calculate the PDF

            // Set the result only for valid elements
            toGLM(ret) = glm::mix(toGLM(ret), pdf, valid);
        }

        return ret;
    }

    __host__ __device__ Vector4f sampleVisibleWavelengths(Vector4f u)
    {
        // Ensure the input is within [0, 1]
        assert(glm::all(glm::greaterThanEqual(toGLM(u), glmZero()) && glm::lessThanEqual(toGLM(u), glmOne())));

        // Sample wavelengths using the given formula
        Vector4f sampled = {fromGLM(538.f - 138.888889f * glm::atanh(0.85691062f - 1.82750197f * toGLM(u)))};

        return sampled;
    }
} // namespace dmt::sampling
