#include "platform-os-utils.h"

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

    } // namespace win32
#endif
} // namespace dmt
