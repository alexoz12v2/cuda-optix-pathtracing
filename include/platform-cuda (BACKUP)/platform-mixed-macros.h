#pragma once

#include "dmtmacros.h"

#if defined(DMT_PLATFORM_MIXED_SHARED)

#if defined(DMT_PLATFORM_MIXED_EXPORTS)
#define DMT_PLATFORM_MIXED_API DMT_API_EXPORT
#else
#define DMT_PLATFORM_MIXED_API DMT_API_IMPORT
#endif

#else

#define DMT_PLATFORM_MIXED_API

#endif