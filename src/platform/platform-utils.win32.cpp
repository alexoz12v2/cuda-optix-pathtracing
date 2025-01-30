#include "platform-utils.h"

#include "platform-os-utils.win32.h"

#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <errhandlingapi.h>
#include <fileapi.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#undef max
#undef min

namespace dmt::os {
    uint64_t processId()
    {
        uint64_t const ret = static_cast<uint64_t>(GetCurrentProcessId());
        return ret;
    }

    uint64_t threadId()
    {
        uint64_t const ret = static_cast<uint64_t>(GetCurrentThreadId());
        return ret;
    }

    // not exported utils --------------------------------------------------------------------------------------------
    void* reserveVirtualAddressSpace(size_t size)
    {
        void* address = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
        return address; // to check whether it is different than nullptr
    }

    size_t systemAlignment()
    {
        SYSTEM_INFO sysInfo{};
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwAllocationGranularity);
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
        void* committed = VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE);
        return committed != nullptr;
    }

    bool freeVirtualAddressSpace(void* address, size_t size) // true if success
    {
        return VirtualFree(address, 0, MEM_RELEASE);
    }

    void decommitPage(void* pageAddress, size_t pageSize) { VirtualFree(pageAddress, pageSize, MEM_DECOMMIT); }

    void* allocate(size_t _bytes, size_t _align) { return _aligned_malloc(_bytes, _align); }

    void deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align) { _aligned_free(ptr); }

    std::vector<std::pair<std::u8string, std::u8string>> getEnv()
    {
        std::vector<std::pair<std::u8string, std::u8string>> vec;
        LPWCH                                                envStrings = GetEnvironmentStringsW();
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

        return vec;
    }

} // namespace dmt::os