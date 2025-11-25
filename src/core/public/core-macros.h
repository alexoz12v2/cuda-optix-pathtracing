#ifndef DMT_CORE_PUBLIC_CORE_MACROS_H
#define DMT_CORE_PUBLIC_CORE_MACROS_H

#include "platform/dmtmacros.h"

#if defined(DMT_CORE_SHARED)

    #if defined(DMT_CORE_EXPORTS)
        #define DMT_CORE_API DMT_API_EXPORT
    #else
        #define DMT_CORE_API DMT_API_IMPORT
    #endif

#else

    #define DMT_CORE_API

#endif
#endif // DMT_CORE_PUBLIC_CORE_MACROS_H
