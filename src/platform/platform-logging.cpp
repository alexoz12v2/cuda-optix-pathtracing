module;

#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <source_location>
#include <string_view>
#include <utility>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#if defined(DMT_DEBUG)
#include <backward.hpp>
#endif

#if defined(DMT_OS_LINUX)
#include <unistd.h>
#if defined(_POSIX_ASYNCHRONOUS_IO)
#include <aio.h> // https://man7.org/linux/man-pages/man7/aio.7.html https://www.gnu.org/software/libc/manual/html_node/Asynchronous-I_002fO.html
#endif
#elif defined(DMT_OS_WINDOWS)
#include <windows.h> // should be included exactly once in every translation unit in which you need definitions
#endif

// https://github.com/fmtlib/fmt/blob/master/include/fmt/format.h, line 4153

module platform;

namespace dmt {
    // CircularOStringStream ------------------------------------------------------------------------------------------
    CircularOStringStream::CircularOStringStream() :
    m_buffer{reinterpret_cast<char*>(reserveVirtualAddressSpace(bufferSize))},
    m_pos{0}
    {
        if (!m_buffer || !commitPhysicalMemory(m_buffer, bufferSize))
        {
            std::abort();
        }
    }

    CircularOStringStream::CircularOStringStream(CircularOStringStream&& other) noexcept :
    m_buffer{std::exchange(other.m_buffer, nullptr)},
    m_pos{std::exchange(other.m_pos, 0)}
    {
    }

    CircularOStringStream& CircularOStringStream::operator=(CircularOStringStream&& other) noexcept
    {
        if (this == &other || !m_buffer)
        {
            return *this;
        }

        freeVirtualAddressSpace(m_buffer, bufferSize);
        m_buffer = std::exchange(other.m_buffer, nullptr);
        m_pos    = std::exchange(other.m_pos, 0);
        return *this;
    }

    CircularOStringStream::~CircularOStringStream() noexcept
    {
        if (m_buffer)
        {
            freeVirtualAddressSpace(m_buffer, bufferSize);
        }
    }


    // Method to simulate writing to the buffer
    CircularOStringStream& CircularOStringStream::operator<<(StrBuf const& buf)
    {
        char const* ps  = buf.len < 0 ? buf.buf : buf.str;
        uint32_t    len = buf.len < 0 ? static_cast<uint32_t>(-buf.len) : static_cast<uint32_t>(buf.len);

        if (uint32_t remaining = bufferSize - m_pos; len <= remaining)
        {
            std::memcpy(m_buffer + m_pos, ps, len);
            m_pos += len;
        }
        else
        {
            assert(false && "Log Buffer length exceeded"); // you shouldn't get here
            // Write the part that fits at the end
            std::memcpy(m_buffer + m_pos, ps, remaining);
            // Wrap around and write the rest
            uint32_t remainingToWrite = len - remaining;
            std::memcpy(m_buffer, ps + remaining, remainingToWrite);
            m_buffer[remainingToWrite] = '\0';

            m_pos = remainingToWrite + 1;
        }
        return *this;
    }

    // Method to simulate writing to the buffer
    CircularOStringStream& CircularOStringStream::operator<<(char const c)
    {
        m_buffer[m_pos] = c;
        m_pos           = (m_pos + 1) % bufferSize; // Properly wrap around
        return *this;
    }

    // To reset the buffer, clearing the content
    void CircularOStringStream::clear()
    {
        m_pos       = 0;
        m_buffer[0] = '\0';
    }

    void CircularOStringStream::logInitList(char const* formatStr, std::initializer_list<StrBuf> const& args)
    {
        // Create a va_list to handle the variable arguments
        char const* p = formatStr; // Pointer to traverse the format string

        // Iterate through the format string
        bool escaped = false;
        for (auto const* it = args.begin(); *p != '\0';)
        {
            if (!escaped && it != args.end() && *p == '{' && *(p + 1) == '}')
            {                   // We've found a placeholder "{}"
                *this << *it++; // Insert the corresponding argument in the stream
                p += 2;         // Skip over the placeholder "{}"
            }
            else
            {
                // If not a placeholder, just append the current character to the stream
                *this << *p;
                escaped = *p == '\\';
                ++p;
            }
        }
    }

    // TODO when you remove C printing functions, remake this const and remove \0 termination
    std::string_view CircularOStringStream::str()
    {
        m_buffer[m_pos] = '\0';
        return {m_buffer, m_pos};
    }

/*
 * ---------------------------------------OS SPECIFIC SECTION-------------------------------------------------
 */
#if defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO) // <- required macro
    static_assert(sizeof(aiocb) <= 168);
    struct PaddedAioCb
    {
        aiocb         acb;
        unsigned char padding[sizeof(AioSpace) - sizeof(aiocb)];
    };

    static_assert(sizeof(PaddedAioCb) == sizeof(AioSpace) && alignof(PaddedAioCb) == alignof(AioSpace));
    static_assert(sizeof(PaddedAioCb) == 256 && std::is_trivial_v<PaddedAioCb> && std::is_standard_layout_v<PaddedAioCb>);

    // TODO add proper memory management
    static_assert(std::is_standard_layout_v<LinuxAsyncIOManager> && sizeof(LinuxAsyncIOManager) == asyncIOClassSize);

    LinuxAsyncIOManager::LinuxAsyncIOManager() :
    m_aioQueue(reinterpret_cast<AioSpace*>(std::aligned_alloc(alignof(PaddedAioCb), numAios * sizeof(PaddedAioCb)))),
    m_lines(reinterpret_cast<Line*>(std::aligned_alloc(alignof(Line), numAios * sizeof(Line))))
    {
        if (!m_aioQueue || !m_lines)
            std::abort();

        initAio();
    }

    void LinuxAsyncIOManager::initAio()
    {
        // Initialize the aiocb structures
        for (uint32_t i = 0; i < numAios; ++i)
        {
            PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[i]);
            std::memset(&m_aioQueue[i], 0, sizeof(PaddedAioCb));
            paddedAioCb.acb.aio_fildes     = STDOUT_FILENO;
            paddedAioCb.acb.aio_offset     = 0; // Default to write operation
            paddedAioCb.acb.aio_buf        = reinterpret_cast<volatile void*>(m_lines[i].buf);
            paddedAioCb.acb.aio_lio_opcode = LIO_WRITE; // Default to write operation
            //paddedAioCb.aio_reqprio    =, scheduling priority. Requires additional macros
            paddedAioCb.acb.aio_sigevent.sigev_notify = SIGEV_NONE; // SIGEV_NONE, SIGEV_SIGNAL -> sigev_signo, SIGEV_THREAD -> sigev_notify_attributes, specify thread: (linux only) sigev_notify_thread_id (used only by timers)
            paddedAioCb.acb.aio_lio_opcode = LIO_WRITE; // only used by lio_listio when you schedule multiple operations
        }
    }

    void LinuxAsyncIOManager::cleanup() noexcept
    {
        sync();
        std::free(m_aioQueue);
        std::free(m_lines);
    }

    LinuxAsyncIOManager::LinuxAsyncIOManager(LinuxAsyncIOManager&& other) noexcept :
    m_aioQueue(std::exchange(other.m_aioQueue, nullptr)),
    m_lines(std::exchange(other.m_lines, nullptr))
    {
    }

    LinuxAsyncIOManager::~LinuxAsyncIOManager() noexcept
    {
        cleanup();
    }

    LinuxAsyncIOManager& LinuxAsyncIOManager::operator=(LinuxAsyncIOManager&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            m_aioQueue = std::exchange(other.m_aioQueue, nullptr);
            m_lines    = std::exchange(other.m_lines, nullptr);
        }
        return *this;
    }

    char* LinuxAsyncIOManager::operator[](uint32_t idx)
    {
        return m_lines[idx].buf;
    }

    uint32_t LinuxAsyncIOManager::findFirstFreeBlocking()
    {
        // Ensure we complete previous operations before starting a new one
        // necessary only if we are switching from STDOUT to STDERR
        // syncIfNeeded(fildes); // we only support STDOUT

        // Find an available aiocb slot (simple round-robin or any available slot)
        while (true) // if everything is full, then poll
        {
            for (uint32_t i = 0; i < numAios; ++i)
            {
                PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[i]);
                if (aio_error(&paddedAioCb.acb) != EINPROGRESS)
                {
                    return i;
                }
            }
        }
    }

    bool handleStatus(int32_t status, aiocb& outAio, uint32_t maxAttempts)
    {

        assert(status != EBADF && "Asynchronous write File Descriptor invalid!");
        assert(status != EINVAL && "Invalid `aio_offset` or `aio_reqprio`!");
        uint32_t attempt = 1;
        for (; status == EAGAIN && attempt != maxAttempts; ++attempt)
        {
            status = aio_write(&outAio);
        }

        return attempt == maxAttempts;
    }

    bool LinuxAsyncIOManager::enqueue(uint32_t idx, size_t size)
    {
        // no sync needed as handled by findFirstFreeBlocking
        // the m_lines[idx].buf should be written externally
        // Find an available aiocb slot (simple round-robin or any available slot)
        PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[idx]);
        assert(aio_error(&paddedAioCb.acb) != EINPROGRESS);
        paddedAioCb.acb.aio_nbytes = size;
        int status                 = aio_write(&paddedAioCb.acb);
        return handleStatus(status, paddedAioCb.acb, maxAttempts);
    }

    void LinuxAsyncIOManager::sync() const
    {
        for (uint32_t i = 0; i != LinuxAsyncIOManager::numAios; ++i)
        {
            PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[i]);
            while (aio_error(&paddedAioCb.acb) == EINPROGRESS)
            {
                // busy waiting...
            }
        }
    }


// standard layout to be castable and memcopied from array of bytes + size requirement
#elif defined(DMT_OS_WINDOWS)

    struct OverlappedWrite
    {
        OVERLAPPED overlapped;
    };

    static_assert(sizeof(OverlappedWrite) == sizeof(AioSpace) && alignof(OverlappedWrite) == alignof(AioSpace));
    static_assert(std::is_standard_layout_v<WindowsAsyncIOManager> && sizeof(WindowsAsyncIOManager) == asyncIOClassSize);

    WindowsAsyncIOManager::WindowsAsyncIOManager() :
    m_hStdOut(CreateFile("CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL)
              // GetStdHandle(STD_OUTPUT_HANDLE)
              ),
    m_aioQueue(reinterpret_cast<AioSpace*>(std::malloc(numAios * sizeof(OverlappedWrite)))),
    m_lines(reinterpret_cast<Line*>(std::malloc(numAios * sizeof(Line))))
    {
        if (!m_aioQueue || !m_lines)
            std::abort();

        initAio();
    }

    void WindowsAsyncIOManager::initAio()
    {
        SECURITY_ATTRIBUTES secAttrs{
            .nLength              = sizeof(SECURITY_ATTRIBUTES),
            .lpSecurityDescriptor = nullptr, // you have the same rights/permissions as the user runnning the process
            .bInheritHandle       = true,    // inherited by child processes
        };

        for (uint32_t i = 0; i != numAios; ++i)
        {
            OverlappedWrite& aioStruct = *reinterpret_cast<OverlappedWrite*>(&m_aioQueue[i]);
            aioStruct.overlapped.hEvent = CreateEventA(&secAttrs, /*manualReset*/ true, /*startSignaled*/ true, /*name*/ nullptr);
            assert(aioStruct.overlapped.hEvent != nullptr && "Couldn't create event!");
        }
    }

    char* WindowsAsyncIOManager::operator[](uint32_t idx)
    {
        return m_lines[idx].buf;
    }

    uint32_t WindowsAsyncIOManager::findFirstFreeBlocking()
    {
        // Try to find a free event with polling (timeout = 0)
        int32_t freeIndex = waitForEvents(0, /*waitAll=*/false);
        if (freeIndex >= 0)
        {
            // io operation resets the event specified by the hEvent member of the OVERLAPPED structure to a
            // nonsignaled state when it begins the I/O operation. Therefore, the caller does not need to do that.
            // ResetEvent(m_aioQueue[freeIndex].overlapped.hEvent);
            return static_cast<uint32_t>(freeIndex); // Return immediately if a free event is found
        }

        // If no free event is found, block until one is available
        // https://learn.microsoft.com/en-gb/windows/win32/api/fileapi/nf-fileapi-createfilea?redirectedfrom=MSDN
        // ^ windows actually doesn't support overlapped output to console. Therefore, if no one is free,
        // reset all events and pick the first one, hoping everything is fine :)
        for (uint32_t i = 0; i != numAios; ++i)
        {
            OverlappedWrite& aioStruct = *reinterpret_cast<OverlappedWrite*>(&m_aioQueue[i]);
            ResetEvent(aioStruct.overlapped.hEvent);
        }

        return 0;
    }

    WindowsAsyncIOManager::WindowsAsyncIOManager(WindowsAsyncIOManager&& other) noexcept :
    m_hStdOut(GetStdHandle(STD_OUTPUT_HANDLE)),
    m_aioQueue(std::exchange(other.m_aioQueue, nullptr)),
    m_lines(std::exchange(other.m_lines, nullptr))
    {
        std::memcpy(m_hBuffer, other.m_hBuffer, sizeof(m_hBuffer));
        std::memset(other.m_hBuffer, 0, sizeof(m_hBuffer));
    }

    WindowsAsyncIOManager& WindowsAsyncIOManager::operator=(WindowsAsyncIOManager&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            std::memcpy(m_hBuffer, other.m_hBuffer, sizeof(m_hBuffer));
            std::memset(other.m_hBuffer, 0, sizeof(m_hBuffer));
            m_aioQueue = std::exchange(other.m_aioQueue, nullptr);
            m_lines    = std::exchange(other.m_lines, nullptr);
        }
        return *this;
    }

    void WindowsAsyncIOManager::sync()
    {
        // overlapped not supported,
        // waitForEvents(INFINITE, /*waitForAll*/ true);
    }

    int32_t WindowsAsyncIOManager::waitForEvents(uint32_t timeout, bool waitAll)
    {
        // https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects
        // try to find a free one with polling, ie WaitForSingleObject with timeout = 0
        // to use wait for multiple objects, we need all event objects in a single buffer
        // Prepare the buffer with event handles
        for (uint32_t i = 0; i < numAios; ++i)
        {
            OverlappedWrite& aioStruct = *reinterpret_cast<OverlappedWrite*>(&m_aioQueue[i]);
            std::memcpy(&m_hBuffer[i], &aioStruct.overlapped.hEvent, sizeof(HANDLE));
        }

        // Wait for events
        DWORD ret = WaitForMultipleObjects(numAios, m_hBuffer, waitAll, timeout);

        // Check for errors or abandoned events
        assert((ret < WAIT_ABANDONED_0 || ret > WAIT_ABANDONED_0 + numAios - 1) && "I/O operation aborted? What?");
        assert(ret != WAIT_FAILED && "Wait operation failed. Call GetLastError().");

        if (WAIT_OBJECT_0 <= ret && ret <= WAIT_OBJECT_0 + numAios - 1)
        {
            return ret - WAIT_OBJECT_0; // Return the index of the signaled event
        }

        if (ret == WAIT_TIMEOUT)
        {
            return -1;
        }

        assert(false && "Unexpected state reached in waitForEvents.");
        return -2; // Unreachable, satisfies compiler
    }

    void WindowsAsyncIOManager::cleanup() noexcept
    {
        sync();

        for (uint32_t i = 0; i != numAios; ++i)
        {
            OverlappedWrite& aioStruct = *reinterpret_cast<OverlappedWrite*>(&m_aioQueue[i]);
            CloseHandle(aioStruct.overlapped.hEvent);
        }
        std::free(m_aioQueue);
        std::free(m_lines);
    }

    WindowsAsyncIOManager::~WindowsAsyncIOManager() noexcept
    {
        cleanup();
    }

    bool WindowsAsyncIOManager::enqueue(int32_t idx, size_t size)
    {
        OverlappedWrite& aioStruct = *reinterpret_cast<OverlappedWrite*>(&m_aioQueue[idx]);
        if (WaitForSingleObject(aioStruct.overlapped.hEvent, INFINITE) != WAIT_OBJECT_0)
        {
            return true;
        }

        aioStruct.overlapped.Offset     = 0xFFFF'FFFF;
        aioStruct.overlapped.OffsetHigh = 0xFFFF'FFFF;

        bool started = WriteFile(m_hStdOut, m_lines[idx].buf, size, nullptr, &aioStruct.overlapped);
        //bool started = WriteConsole(m_hStdOut, m_lines[idx].buf, size, nullptr, nullptr);

        DWORD err = GetLastError();
        if (started || err == ERROR_IO_PENDING)
        {
            // io started
            return false;
        }
        else
        {
            // uncomment this if you need to debug duplicate lines
            //char*  buf  = nullptr;
            //size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            //                                 FORMAT_MESSAGE_IGNORE_INSERTS,
            //                             nullptr,
            //                             err,
            //                             MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            //                             (LPSTR)&buf,
            //                             0,
            //                             nullptr);
            //LocalFree(buf);
            return true;
        }
    }

#endif
    /*
     * -----------------------------------END OS SPECIFIC SECTION-------------------------------------------------
     */

    ConsoleLogger::~ConsoleLogger()
    {
        m_IOClassInterface.destructor(m_asyncIOClass);
    }

    void ConsoleLogger::write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
    {
        if (enabled(level))
        {
            std::lock_guard<std::mutex> lock(s_writeMutex);
            std::string_view            date     = getCurrentTimestamp();
            std::string_view            fileName = loc.file_name();
            carveRelativeFileName(fileName);
            logMessageAsync(level, date, fileName, loc.function_name(), loc.line(), stringFromLevel(level), str);
        }
    }

    void ConsoleLogger::write(ELogLevel                            level,
                              std::string_view const&              str,
                              std::initializer_list<StrBuf> const& list,
                              std::source_location const&          loc)
    {
        if (enabled(level))
        {
            std::lock_guard<std::mutex> lock(s_writeMutex);
            std::string_view            date     = getCurrentTimestamp();
            std::string_view            fileName = loc.file_name();
            carveRelativeFileName(fileName);

            m_oss.logInitList(str.data(), list);
            logMessageAsync(level, date, fileName, loc.function_name(), loc.line(), stringFromLevel(level), m_oss.str());
            m_oss.clear();
        }
    }

    // Helper function to format and print the log message
    void ConsoleLogger::logMessage(
        ELogLevel               level,
        std::string_view const& date,
        std::string_view const& fileName,
        std::string_view const& functionName,
        uint32_t                line,
        std::string_view const& levelStr,
        std::string_view const& content)
    {
        std::printf("%s[%s %s:%s:%u] %s <> %s\n%s",
                    logcolor::colorFromLevel(level).data(),
                    date.data(),
                    fileName.data(),
                    functionName.data(),
                    line,
                    levelStr.data(),
                    content.data(),
                    logcolor::reset.data());
    }

    void ConsoleLogger::logMessageAsync(
        ELogLevel               level,
        std::string_view const& date,
        std::string_view const& fileName,
        std::string_view const& functionName,
        uint32_t                line,
        std::string_view const& levelStr,
        std::string_view const& content)
    {
#if (defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO)) || defined(DMT_OS_WINDOWS)
        bool res = m_IOClassInterface.tryAsyncLog(m_asyncIOClass, level, date, fileName, functionName, line, levelStr, content);
        if (res)
        {
            logMessage(level, date, fileName, functionName, line, levelStr, content);
        }
#else
        logMessage(level, date, fileName, functionName, line, levelStr, content);
#endif
    }

    // Helper function to get a relative file name
    void ConsoleLogger::carveRelativeFileName(std::string_view& fullPath)
    {
        static constexpr std::string_view projPath = DMT_PROJ_PATH;
        assert(fullPath.starts_with(projPath) && "the specified file is outside of the project!");
        fullPath.remove_prefix(projPath.size() + 1);
    }

    // Helper function to get the current timestamp
    std::string_view ConsoleLogger::getCurrentTimestamp()
    {
        std::time_t now     = std::time(nullptr);
        std::tm     tstruct = *std::localtime(&now);
        std::strftime(m_timestampBuf, sizeof(m_timestampBuf), "%Y-%m-%d.%X", &tstruct);
        return m_timestampBuf;
    }

    ConsoleLogger::ConsoleLogger(ConsoleLogger&& other) : BaseLogger<ConsoleLogger>(other.m_level)
    {
        std::lock_guard lock{s_writeMutex};
        stealResourcesFrom(std::move(other));
    }

    ConsoleLogger& ConsoleLogger::operator=(ConsoleLogger&& other)
    {
        if (this != &other)
        {
            std::lock_guard lock{s_writeMutex};

            // destroy my manager
            m_IOClassInterface.destructor(m_asyncIOClass);
            stealResourcesFrom(std::move(other));
        }
        return *this;
    }

    void ConsoleLogger::stealResourcesFrom(ConsoleLogger&& other)
    {
        // m_oss doesn't need to be moved as it gets reset every time the lock is released
        // m_timestampBuf[timestampMax] doesn't need any special handling as it is a local buffer whose state is useful within a single `write` call
        memcpy(m_asyncIOClass, other.m_asyncIOClass, sizeof(m_asyncIOClass));
        std::construct_at<NullAsyncIOManager>(reinterpret_cast<NullAsyncIOManager*>(other.m_asyncIOClass));
        m_IOClassInterface = std::exchange(other.m_IOClassInterface, Table{});
    }

    static_assert(LogDisplay<ConsoleLogger>);

    // LoggingContext -------------------------------------------------------------------------------------------------
    void LoggingContext::dbgTraceStackTrace()
    {
#if defined(DMT_DEBUG)
        if (traceEnabled())
        {
            trace("Printing StackTrace");
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#endif
    }

    void LoggingContext::dbgErrorStackTrace()
    {
#if defined(DMT_DEBUG)
        if (errorEnabled())
        {
            error("Printing StackTrace");
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#endif
    }


} // namespace dmt