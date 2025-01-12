#pragma once

#include "dmtmacros.h"
#include <platform/platform-macros.h>

#include <utility> // std::min

#include <cmath> // std::min
#include <cstdint>
#include <cstring> // std::memcpy

#if defined(DMT_OS_WINDOWS)
//#if defined(DMT_COMPILER_MSVC)
//#pragma warning(disable : 4005 5106) // SHut SAL macro redefinition up
//#endif
// clang-format off
#pragma comment(lib, "mincore")
#define NO_SAL
#include <Windows.h>
#include <AclAPI.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#undef min
#undef max
// clang-format on
#elif defined(DMT_OS_LINUX)
#include <sys/mman.h> // mmap, munmap
#endif

// module-private bits of functionality. Should not be exported by the primary interface unit
// note: since this is to be visible to all implementation units, it cannot be put into a .cpp file, as
// implementation units are not linked together. It needs to stay here. It should not be included in the
// binary interface, hence it is fine
namespace dmt {
#if defined(DMT_OS_WINDOWS)
    namespace win32 {

        uint32_t getLastErrorAsString(char* buffer, uint32_t maxSize);

        constexpr bool luidCompare(LUID const& luid0, LUID const& luid1)
        {
            return luid0.HighPart == luid1.HighPart && luid1.LowPart == luid0.LowPart;
        }

    } // namespace win32
#elif defined(DMT_OS_LINUX)
    namespace linux {
    }
#endif
} // namespace dmt
