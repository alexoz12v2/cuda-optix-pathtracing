#pragma once

#include "dmtmacros.h"

#if defined(DMT_MIDDLEWARE_SHARED)

#if defined(DMT_MIDDLEWARE_EXPORTS)
#define DMT_MIDDLEWARE_API DMT_API_EXPORT
#else
#define DMT_MIDDLEWARE_API DMT_API_IMPORT
#endif

#else

#define DMT_MIDDLEWARE_API

#endif