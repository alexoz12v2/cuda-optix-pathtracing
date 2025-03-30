#include "platform-os-utils.win32.h"

#include <bit>
#include <memory>

#include <cassert>

#include <strsafe.h>

namespace dmt::os::win32 {
    uint32_t utf16le_From_utf8(char8_t const* DMT_RESTRICT _u8str,
                               uint32_t                    _u8NumBytes,
                               wchar_t* DMT_RESTRICT       _mediaBuf,
                               uint32_t                    _mediaMaxBytes,
                               wchar_t* DMT_RESTRICT       _outBuf,
                               uint32_t                    _maxBytes,
                               uint32_t*                   _outBytesWritten)
    {
        if (_outBytesWritten)
            *_outBytesWritten = static_cast<uint32_t>(sizeof(wchar_t));
        int res = MultiByteToWideChar(CP_UTF8,
                                      MB_ERR_INVALID_CHARS,
                                      std::bit_cast<char const*>(_u8str),
                                      static_cast<int>(_u8NumBytes),
                                      _mediaBuf,
                                      _mediaMaxBytes / sizeof(wchar_t));
        // TODO if (res == 0) errror, else number is positive
        assert(res >= 0);

        int estimatedSize = NormalizeString(::NormalizationC, _mediaBuf, res, nullptr, 0);
        if (estimatedSize > (_maxBytes >> 1)) // means divided by sizeof(wchar_t)
            assert(false);                    // TODO better
        int actualLength = NormalizeString(::NormalizationC, _mediaBuf, res, _outBuf, _maxBytes >> 1);

        if (_outBytesWritten)
            *_outBytesWritten *= actualLength;
        return actualLength;
    }

    std::unique_ptr<wchar_t[]> quickUtf16leFrom(char const* prefix, char const* str)
    {
        uint32_t const             len      = static_cast<uint32_t>(std::strlen(prefix) + std::strlen(str));
        uint32_t const             numChars = len + 16;
        std::unique_ptr<wchar_t[]> normBuf  = std::make_unique<wchar_t[]>(numChars);
        std::unique_ptr<wchar_t[]> midBuf   = std::make_unique<wchar_t[]>(numChars - 15);

        uint32_t numOutChars = utf16le_From_utf8(std::bit_cast<char8_t const*>(str),
                                                 static_cast<uint32_t>(std::strlen(prefix)),
                                                 midBuf.get(),
                                                 (numChars - 15) << 1,
                                                 normBuf.get(),
                                                 numChars << 1,
                                                 nullptr);
        numOutChars += utf16le_From_utf8(std::bit_cast<char8_t const*>(str),
                                         static_cast<uint32_t>(std::strlen(str)),
                                         midBuf.get(),
                                         (numChars - 15) << 1,
                                         normBuf.get() + numChars,
                                         numChars << 1,
                                         nullptr);
        normBuf[numOutChars] = L'\0';
        return normBuf;
    }

    void errorExit(wchar_t const* msg)
    {
        wchar_t*       lpMsgBuf;
        wchar_t*       lpDisplayBuf;
        wchar_t const* display;
        DWORD          dw = GetLastError();

        FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL,
                       dw,
                       MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       reinterpret_cast<wchar_t*>(&lpMsgBuf),
                       0,
                       NULL);

        DWORD const numBytes = (wcslen(lpMsgBuf) + wcslen(msg) + 40) * sizeof(wchar_t);
        lpDisplayBuf         = reinterpret_cast<wchar_t*>(LocalAlloc(LMEM_ZEROINIT, numBytes));

        if (lpDisplayBuf)
        {
            StringCchPrintfW(lpDisplayBuf,
                             LocalSize(lpDisplayBuf) / sizeof(wchar_t),
                             L"%s failed with error %d: %s",
                             msg,
                             dw,
                             lpMsgBuf);
            display = lpDisplayBuf;
        }
        else
            display = msg;
        MessageBoxW(NULL, display, L"Error", MB_OK);

        LocalFree(lpMsgBuf);
        LocalFree(lpDisplayBuf);
        ExitProcess(1);
    }

    // TODO BETTER
    uint32_t getLastErrorAsString(char* buffer, uint32_t maxSize)
    {
        //Get the error message ID, if any.
        DWORD errorMessageID = ::GetLastError();
        if (errorMessageID == 0)
        {
            buffer[0] = '\n';
            return 0;
        }
        else
        {

            LPSTR messageBuffer = nullptr;

            //Ask Win32 to give us the string version of that message ID.
            //The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                             FORMAT_MESSAGE_IGNORE_INSERTS,
                                         NULL,
                                         errorMessageID,
                                         MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                         (LPSTR)&messageBuffer,
                                         0,
                                         NULL);

            //Copy the error message into a std::string.
            size_t actual = std::min(static_cast<size_t>(maxSize - 1), size);
            std::memcpy(buffer, messageBuffer, actual);
            buffer[actual] = '\0';

            //Free the Win32's string's buffer.
            LocalFree(messageBuffer);
            return actual;
        }
    }

    std::pmr::wstring utf16FromUtf8(std::string_view mbStr, std::pmr::memory_resource* resource)
    {
        using namespace std::string_literals;
        std::pmr::wstring result{resource};
        int               sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, mbStr.data(), mbStr.length(), nullptr, 0);
        if (sizeNeeded > 0)
        {
            std::unique_ptr<wchar_t[]> buf = std::make_unique<wchar_t[]>(sizeNeeded + 1);
            buf[sizeNeeded]                = L'\0';
            MultiByteToWideChar(CP_UTF8, 0, mbStr.data(), mbStr.length(), buf.get(), sizeNeeded);
            result.append(buf.get());
        }

        return result;
    }

    std::pmr::string utf8FromUtf16(std::wstring_view wideStr, std::pmr::memory_resource* resource)
    {
        using namespace std::string_literals;
        std::pmr::string result{resource};
        int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wideStr.data(), wideStr.length(), nullptr, 0, nullptr, nullptr);
        if (sizeNeeded > 0)
        {
            std::unique_ptr<char[]> buf = std::make_unique<char[]>(sizeNeeded + 1);
            buf[sizeNeeded]             = '\0';
            WideCharToMultiByte(CP_UTF8, 0, wideStr.data(), wideStr.length(), buf.get(), sizeNeeded, nullptr, nullptr);
            result.append(buf.get());
        }
        return result;
    }
} // namespace dmt::os::win32
