#pragma once

#include "dmtmacros.h"

#if defined(DMT_CUDA_WRAPPERS_SHARED) && !defined(__NVCC__)

#  if defined(DMT_CUDA_WRAPPERS_EXPORTS)
#    define DMT_CUDA_WRAPPERS_API DMT_API_EXPORT
#  else
#    define DMT_CUDA_WRAPPERS_API DMT_API_IMPORT
#  endif

#else

#  define DMT_CUDA_WRAPPERS_API

#endif