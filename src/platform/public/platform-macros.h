#ifndef DMT_PLATFORM_PUBLIC_PLATFORM_MACROS_H
#define DMT_PLATFORM_PUBLIC_PLATFORM_MACROS_H

#include "dmtmacros.h"

#if defined(DMT_PLATFORM_SHARED) && !defined(__NVCC__)

    #if defined(DMT_PLATFORM_EXPORTS)
        #define DMT_PLATFORM_API DMT_API_EXPORT
    #else
        #define DMT_PLATFORM_API DMT_API_IMPORT
    #endif

#else

    #define DMT_PLATFORM_API

#endif
#endif // DMT_PLATFORM_PUBLIC_PLATFORM_MACROS_H
