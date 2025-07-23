#pragma once

#include "cudautils/cudautils-macro.h"

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_MATERIAL_IMPL)
    #include "cudautils-material.cu"
#endif
