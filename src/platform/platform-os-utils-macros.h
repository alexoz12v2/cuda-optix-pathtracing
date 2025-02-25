#pragma once

#include "dmtmacros.h"

#if defined(DMT_PLATFORM_OS_UTILS_SHARED) && !defined(__NVCC__)

#if defined(DMT_PLATFORM_OS_UTILS_EXPORTS)
#define DMT_PLATFORM_OS_UTILS_API DMT_API_EXPORT
#else
#define DMT_PLATFORM_OS_UTILS_API DMT_API_IMPORT
#endif

#else

#define DMT_PLATFORM_OS_UTILS_API

#endif
