#pragma once

#include "cudautils/cudautils-macro.h"

#include <cudautils/cudautils-vecmath.h>

// u = random uniformly distributed fp32 number between 0 and 1
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

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_NUMBERS_IMPL)
#include "cudautils-numbers.cu"
#endif
