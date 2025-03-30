#include "cudautils-numbers.h"

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
