//module;
#define DMT_INTERFACE_AS_HEADER
#undef DMT_NEEDS_MODULE
#include "platform-utils.h"

#include "platform-os-utils.h"

#include <array>
#include <bit>
#include <concepts>
#include <shared_mutex>
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
#elif defined(DMT_OS_LINUX)
#include <unistd.h>
#endif

//module platform;

namespace dmt {
    uint64_t processId()
    {
        uint64_t ret = 0;
#if defined(DMT_OS_WINDOWS)
        DWORD const pid = GetCurrentProcessId();
        ret             = static_cast<uint64_t>(pid);
#elif defined(DMT_OS_LINUX)
        pid_t const pid = getpid();
        ret             = static_cast<uint64_t>(pid);
#endif
        return ret;
    }

    uint64_t threadId()
    {
        uint64_t ret = 0;
#if defined(DMT_OS_WINDOWS)
        ret = static_cast<uint64_t>(GetCurrentThreadId());
#elif defined(DMT_OS_LINUX)
        ret             = static_cast<uint64_t>(gettid());
#endif
        return ret;
    }

    // not exported utils --------------------------------------------------------------------------------------------
    void* reserveVirtualAddressSpace(size_t size)
    {
#if defined(DMT_OS_WINDOWS)
        void* address = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
        return address; // to check whether it is different than nullptr
#elif defined(DMT_OS_LINUX)
        void* address   = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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

    std::vector<std::pair<std::u8string, std::u8string>> getEnv()
    {
        std::vector<std::pair<std::u8string, std::u8string>> vec;
#if defined(DMT_OS_WINDOWS)
        LPWCH envStrings = GetEnvironmentStringsW();
        if (!envStrings)
            return vec;

        LPWCH current = envStrings;
        while (*current)
        {
            std::wstring_view wideEntry{current};
            size_t            pos = wideEntry.find(L'=');
            if (pos != std::wstring_view::npos)
            {
                std::wstring_view wideName  = wideEntry.substr(0, pos);
                std::wstring_view wideValue = wideEntry.substr(pos + 1);

                // TODO: refactor this outside
                if (!wideEntry.starts_with(L'='))
                {
                    std::u8string name  = win32::utf8FromUtf16(wideName);
                    std::u8string value = win32::utf8FromUtf16(wideValue);
                    vec.emplace_back(name, value);
                }
            }

            current += wcslen(current) + 1;
        }
        FreeEnvironmentStringsW(envStrings);
#elif defined(DMT_OS_LINUX)
#endif
        return vec;
    }

    namespace detail {
        std::map<uint64_t, CtxCtrlBlock> g_ctxMap;
        std::shared_mutex                g_slk;
    } // namespace detail

} // namespace dmt
