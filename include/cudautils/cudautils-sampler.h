#pragma once

#include "cudautils/cudautils-macro.h"

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_SAMPLER_IMPL)
    #include "cudautils-sampler.cu"
#endif
