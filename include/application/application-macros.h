#pragma once

#include "dmtmacros.h"

#if defined(DMT_APPLICATION_SHARED) && !defined(__NVCC__)

    #if defined(DMT_APPLICATION_EXPORTS)
        #define DMT_APPLICATION_API DMT_API_EXPORT
    #else
        #define DMT_APPLICATION_API DMT_API_IMPORT
    #endif

#else

    #define DMT_APPLICATION_API

#endif