//module;
#define DMT_INTERFACE_AS_HEADER
#undef DMT_NEEDS_MODULE
#include "platform-utils.h"

#include "platform-os-utils.h"

#include <array>
#include <bit>
#include <concepts>
#include <limits>

#include <cassert>

#if defined(DMT_OS_WINDOWS)
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <errhandlingapi.h>
#include <fileapi.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#undef max
#undef min
#endif

//module platform;

namespace dmt {

    // not exported utils --------------------------------------------------------------------------------------------
    void* reserveVirtualAddressSpace(size_t size)
    {
#if defined(DMT_OS_WINDOWS)
        void* address = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
        return address; // to check whether it is different than nullptr
#elif defined(DMT_OS_LINUX)
        void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (address == MAP_FAILED)
        {
            return nullptr;
        }
        return address;
#endif
    }

    size_t systemAlignment()
    {
#if defined(DMT_OS_WINDOWS)
        SYSTEM_INFO sysInfo{};
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwAllocationGranularity);
#elif defined(DMT_OS_LINUX)
        // TODO
        return 0;
#endif
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
#if defined(DMT_OS_WINDOWS)
        void* committed = VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE);
        return committed != nullptr;
#elif defined(DMT_OS_LINUX)
        int result = mprotect(address, size, PROT_READ | PROT_WRITE);
        return result == 0;
#endif
    }

    bool freeVirtualAddressSpace(void* address, size_t size) // true if success
    {
#if defined(DMT_OS_WINDOWS)
        return VirtualFree(address, 0, MEM_RELEASE);
#elif defined(DMT_OS_LINUX)
        return !munmap(address, size);
#endif
    }

    void decommitPage(void* pageAddress, size_t pageSize)
    {
#if defined(DMT_OS_WINDOWS)
        VirtualFree(pageAddress, pageSize, MEM_DECOMMIT);
#elif defined(DMT_OS_LINUX)
        mprotect(pageAddress, pageSize, PROT_NONE);
        madvise(pageAddress, pageSize, MADV_DONTNEED); // Optional: Release physical memory
#endif
    }

} // namespace dmt
