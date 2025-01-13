#include "platform-os-utils.h"

#include <memory>

namespace dmt {
#if defined(DMT_OS_WINDOWS)
    namespace win32 {
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

        // TODO if used beyond debugging, write a version which uses our memory systems
        std::u8string utf8FromUtf16(std::wstring_view wideStr)
        {
            using namespace std::string_literals;
            int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wideStr.data(), wideStr.length(), nullptr, 0, nullptr, nullptr);
            if (sizeNeeded == 0)
            {
                return u8""s;
            }
            std::unique_ptr<char[]> buf = std::make_unique<char[]>(sizeNeeded + 1);
            buf[sizeNeeded]             = '\0';
            WideCharToMultiByte(CP_UTF8, 0, wideStr.data(), wideStr.length(), buf.get(), sizeNeeded, nullptr, nullptr);
            return std::u8string(reinterpret_cast<char8_t const*>(buf.get()));
        }

    } // namespace win32
#endif
} // namespace dmt
