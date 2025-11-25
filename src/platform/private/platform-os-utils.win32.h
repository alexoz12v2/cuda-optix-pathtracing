#ifndef DMT_PLATFORM_PRIVATE_PLATFORM_OS_UTILS_WIN32_H
#define DMT_PLATFORM_PRIVATE_PLATFORM_OS_UTILS_WIN32_H

#include "platform/platform-macros.h"

#include <memory>
#include <utility> // std::min
#include <string>
#include <string_view>

#include <cmath> // std::min
#include <cstdint>
#include <cstring> // std::memcpy

#pragma comment(lib, "mincore")
#include <Windows.h>
#include <AclAPI.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#undef min // I am compiling with NOMINMAX, so technically we can remove that
#undef max

// module-private bits of functionality. Should not be exported by the primary interface unit
// note: since this is to be visible to all implementation units, it cannot be put into a .cpp file, as
// implementation units are not linked together. It needs to stay here. It should not be included in the
// binary interface, hence it is fine
namespace dmt::os::win32 {
    uint32_t getLastErrorAsString(char* buffer, uint32_t maxSize);

    constexpr bool luidCompare(LUID const& luid0, LUID const& luid1)
    {
        return luid0.HighPart == luid1.HighPart && luid1.LowPart == luid0.LowPart;
    }

    std::pmr::wstring utf16FromUtf8(std::string_view           mbStr,
                                    std::pmr::memory_resource* resource = std::pmr::get_default_resource());
    std::pmr::string  utf8FromUtf16(std::wstring_view          wideStr,
                                    std::pmr::memory_resource* resource = std::pmr::get_default_resource());

    uint32_t utf16le_From_utf8(char const* DMT_RESTRICT _u8str,
                               uint32_t                 _u8NumBytes,
                               wchar_t* DMT_RESTRICT    _mediaBuf,
                               uint32_t                 _mediaMaxBytes,
                               wchar_t* DMT_RESTRICT    _outBuf,
                               uint32_t                 _maxBytes,
                               uint32_t*                _outBytesWritten);

    /** requires a nul terminated string, returns a nul terminated string */
    std::unique_ptr<wchar_t[]> quickUtf16leFrom(char const* prefix, char const* str);

    template <typename F>
        requires std::is_invocable_r_v<bool, F, std::wstring_view>
    inline bool constructPathAndDo(std::wstring_view path, F&& func)
    {
        if (DWORD length = GetFullPathNameW(path.data(), 0, nullptr, nullptr); length != 0)
        {
            std::unique_ptr<wchar_t[]> buffer = std::make_unique<wchar_t[]>(length + 4);
            buffer[0] = L'\\', buffer[1] = L'\\', buffer[2] = L'?', buffer[3] = L'\\';
            DWORD numChars = GetFullPathNameW(path.data(), length, buffer.get() + 4, nullptr);
            if (numChars == length - 1)
            {
                std::wstring_view fileName = (buffer[4] == L'\\' && buffer[5] == L'\\' && buffer[6] == L'?' &&
                                              buffer[7] == L'\\')
                                                 ? &buffer[4]
                                                 : buffer.get();
                return func(fileName);
            }
        }
        return false;
    }

    DMT_FORCEINLINE DWORD peekLastError()
    {
        DWORD const ret = GetLastError();
        SetLastError(ret);
        return ret;
    }

    [[noreturn]] void errorExit(wchar_t const* msg);
} // namespace dmt::os::win32
#endif // DMT_PLATFORM_PRIVATE_PLATFORM_OS_UTILS_WIN32_H
