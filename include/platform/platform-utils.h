#pragma once

#include "dmtmacros.h"

#include <concepts>
#include <source_location>
#include <string_view>

#include <cmath>
#include <cstdint>
#include <cstdio>

#if defined(DMT_OS_WINDOWS)
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#elif defined(DMT_OS_LINUX)
#endif

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-logging.h>
#else
import <platform/platform-logging.h>;
#endif

DMT_MODULE_EXPORT dmt {
    /**
     * Convenience pointer type to pass around systems of the application to get an enriched/targeted
     * interface to the platform
     * @warning I don't like the fact that we are redunding the logger interface, in particular the
     * redundant `m_level`
     */
    class PlatformContext : public BaseLogger<PlatformContext>
    {
    public:
        // -- Types --
        /**
         * Required stuff from the `BaseLogger`
         */
        struct Traits
        {
            static constexpr ELogDisplay displayType = ELogDisplay::Forward;
        };

        /**
         * Function pointer table stored elsewhere. It is supposed to outlive the context.
         * Meant for functions which are not called often, therefore they can afford the
         * double indirection
         */
        struct Table
        {
            void (*changeLevel)(void* data, ELogLevel level) = [](void* data, ELogLevel level) {};
        };

        /**
         * Function pointer table stored inline here. Meant for functions which are called
         * often
         */
        struct InlineTable
        {
            void (*write)(void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc) =
                [](void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc) {};
            void (*writeArgs)(void*                                data,
                              ELogLevel                            level,
                              std::string_view const&              str,
                              std::initializer_list<StrBuf> const& list,
                              std::source_location const&          loc) =
                [](void*                                data,
                   ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc) {};
            bool (*checkLevel)(void* data, ELogLevel level) = [](void* data, ELogLevel level) { return false; };
        };

        PlatformContext(void* data, Table const* pTable, InlineTable const& inlineTable) :
        m_table(pTable),
        m_inlineTable(inlineTable),
        m_data(data)
        {
        }

        /**
         * Setter for the `m_level`
         * @param level new level
         * @warning Purposefully name hiding the `BaseLogger`
         */
        void setLevel(ELogLevel level)
        {
            m_table->changeLevel(m_data, level);
        }

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str string to output
         * @param loc location of the log
         */
        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
        {
            m_inlineTable.write(m_data, level, str, loc);
        }

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str format string
         * @param list list of arguments which will be used to create the final string
         * @param loc location of the log
         */
        void write(ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc)
        {
            m_inlineTable.writeArgs(m_data, level, str, list, loc);
        }

        /**
         * CRTP overridden function to check if the true underlying logger is enabled on the log level
         * @param level log level requested
         * @return bool signaling whether the requested log level is enabled
         */
        bool enabled(ELogLevel level)
        {
            return m_inlineTable.checkLevel(m_data, level);
        }

    private:
        /**
         * table of infrequent functions offloaded, stored stored elsewhere
         */
        Table const* m_table;

        /**
         * Table of frequent functions stored here
         */
        InlineTable m_inlineTable;

        /**
         * Pointer to a type erased class, which can be casted back in the function pointer
         * functions, like `Platform`
         */
        void* m_data = nullptr;
    };
}; // namespace dmt

    // module-private bits of functionality. Should not be exported by the primary interface unit
    // note: since this is to be visible to all implementation units, it cannot be put into a .cpp file, as
    // implementation units are not linked together. It needs to stay here. It should not be included in the
    // binary interface, hence it is fine
    namespace dmt {
        template <std::integral I>
        constexpr I ceilDiv(I num, I den)
        {
            return (num + den - 1) / den;
        }

#if defined(DMT_OS_WINDOWS)
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

        void decommitPage(void* pageAddress, size_t pageSize)
        {
            VirtualFree(pageAddress, pageSize, MEM_DECOMMIT);
        }

        namespace win32
        {

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
#undef min
                size_t actual = std::min(static_cast<size_t>(maxSize - 1), size);
                std::memcpy(buffer, messageBuffer, actual);
                buffer[actual] = '\0';

                //Free the Win32's string's buffer.
                LocalFree(messageBuffer);
                return actual;
            }
        }

        constexpr bool luidCompare(LUID const& luid0, LUID const& luid1)
        {
            return luid0.HighPart == luid1.HighPart && luid1.LowPart == luid0.LowPart;
        }

        } // namespace win32
#elif defined(DMT_OS_LINUX)
        void* reserveVirtualAddressSpace(size_t size)
        {
            void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (address == MAP_FAILED)
            {
                return nullptr;
            }
            return address;
        }

        bool commitPhysicalMemory(void* address, size_t size)
        {
            int result = mprotect(address, size, PROT_READ | PROT_WRITE);
            return result == 0;
        }

        bool freeVirtualAddressSpace(void* address, size_t size)
        {
            return !munmap(address, size);
        }

        void decommitPage(void* pageAddress, size_t pageSize)
        {
            mprotect(pageAddress, pageSize, PROT_NONE);
            madvise(pageAddress, pageSize, MADV_DONTNEED); // Optional: Release physical memory
        }

        size_t systemAlignment()
        {
            // TODO
            return 0;
        }

        namespace linux
        {
        }
#endif
    } // namespace dmt
