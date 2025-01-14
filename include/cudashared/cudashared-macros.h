#pragma once

#include "dmtmacros.h"

#if defined(DMT_CUDASHARED_SHARED)

#if defined(DMT_CUDASHARED_EXPORTS)
#define DMT_CUDASHARED_API DMT_API_EXPORT
#else
#define DMT_CUDASHARED_API DMT_API_IMPORT
#endif

#else

#define DMT_CUDASHARED_API

#endif