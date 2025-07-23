#pragma once

#include "dmtmacros.h"

// Workaround to make shared module among __CUDA_ARCH__ and host code with dll when with host
// ASSUMES EXPORT MACRO NAME
#if !defined(DMT_CORE_API) && !defined(__NVCC__) && !defined(__CUDA_ARCH__)
    #error "dfjklsdajflkdsjlkfajldsa"
#endif
