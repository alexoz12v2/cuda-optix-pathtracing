#pragma once

#include "dmtmacros.h"

// Workaround to make shared module among __CUDA_ARCH__ and host code with dll when with host
// ASSUMES EXPORT MACRO NAME
#if !defined(__NVCC__) || !defined(__CUDA_ARCH__)
    #if !defined(DMT_CORE_API)
        #include "core/core-macros.h"
    #endif
#else
    #define DMT_CORE_API
#endif
