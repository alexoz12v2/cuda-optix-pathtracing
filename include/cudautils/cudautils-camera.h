#pragma once

#include "cudautils/cudautils-macro.h"

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_CAMERA_IMPL)
    #include "cudautils-camera.cu"
#endif