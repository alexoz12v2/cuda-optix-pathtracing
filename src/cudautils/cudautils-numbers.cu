#include "cudautils-numbers.h"

namespace dmt {
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
