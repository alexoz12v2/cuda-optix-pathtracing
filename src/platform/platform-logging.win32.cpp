#include "platform-logging.h"

#include <windows.h> // should be included exactly once in every translation unit in which you need definitions

namespace dmt {
    struct OverlappedWrite
    {
        OVERLAPPED overlapped;
    };

    static_assert(sizeof(OverlappedWrite) == sizeof(AioSpace) && alignof(OverlappedWrite) == alignof(AioSpace));
    static_assert(std::is_standard_layout_v<WindowsAsyncIOManager> && sizeof(WindowsAsyncIOManager) == asyncIOClassSize);

    WindowsAsyncIOManager::WindowsAsyncIOManager() :
    m_hStdOut(CreateFileW(L"CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL)
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

    char* WindowsAsyncIOManager::operator[](uint32_t idx) { return m_lines[idx].buf; }

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

    WindowsAsyncIOManager::~WindowsAsyncIOManager() noexcept { cleanup(); }

    bool WindowsAsyncIOManager::enqueue(int32_t idx, size_t size)
    {
        // TODO it seems that, in a multithreaded environment, the logger breaks if you don't wait
        // fix this
        waitForEvents(INFINITE, true);
        OverlappedWrite& aioStruct = *reinterpret_cast<OverlappedWrite*>(&m_aioQueue[idx]);
        //if (WaitForSingleObject(aioStruct.overlapped.hEvent, INFINITE) != WAIT_OBJECT_0)
        //{
        //    return true;
        //}

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

    // LOGGING 2.0 ----------------------------------------------------------------------------------------------------

    void uft16le_From_utf8(char8_t const* DMT_RESTRICT _u8str,
                           uint32_t                    _u8NumBytes,
                           wchar_t* DMT_RESTRICT       _mediaBuf,
                           uint32_t                    _mediaMaxBytes,
                           wchar_t* DMT_RESTRICT       _outBuf,
                           uint32_t                    _maxBytes,
                           uint32_t*                   _outBytesWritten);
    struct LoggerData
    {
        LoggerData() :
        hStdOut(CreateFileW(L"CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr))
        {
            // TODO Check result
            // write at the end ?
            //overlapped.Offset     = 0xFFFF'FFFF;
            //overlapped.OffsetHigh = 0xFFFF'FFFF;
            overlapped.hEvent = CreateEventW(nullptr,  // default security attributes
                                             true,     // manual reset
                                             true,     // start signaled
                                             nullptr); // unnamed event
            // TODO if overlapped.hEvent is null
        }
        LoggerData(LoggerData const&)                = delete;
        LoggerData(LoggerData&&) noexcept            = delete;
        LoggerData& operator=(LoggerData const&)     = delete;
        LoggerData& operator=(LoggerData&&) noexcept = delete;
        ~LoggerData() noexcept
        {
            // TODO if failed crash
            WaitForSingleObject(overlapped.hEvent, INFINITE);
            CloseHandle(overlapped.hEvent);
            CloseHandle(hStdOut);
        }

        void writeAsync(char8_t const* _str, DWORD _numBytes)
        { // if wait failed crash or abandon logging
            DWORD query = WaitForSingleObject(overlapped.hEvent, 0);
            if (query == WAIT_TIMEOUT)
                query = WaitForSingleObject(overlapped.hEvent, INFINITE);
            // convert UTF-8 to UTF-16 LE
            uint32_t bytes = 0;
            uft16le_From_utf8(_str, _numBytes, buffer, 2048, normalizedBuffer, 2048, &bytes);
            // TODO if fails, crash?
            WriteFileEx(hStdOut, normalizedBuffer, bytes, &overlapped, nullptr);
        }

        OVERLAPPED overlapped{};
        HANDLE     hStdOut = INVALID_HANDLE_VALUE;
        wchar_t    buffer[2048]{};
        wchar_t    normalizedBuffer[2048]{};
    };

    // TODO move to OS specific utils
    void uft16le_From_utf8(char8_t const* DMT_RESTRICT _u8str,
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
                                      MB_PRECOMPOSED | MB_ERR_INVALID_CHARS,
                                      std::bit_cast<char const*>(_u8str),
                                      static_cast<int>(_u8NumBytes),
                                      _mediaBuf,
                                      _mediaMaxBytes / sizeof(wchar_t));
        // TODO if (res == 0) errror, else number is positive
        assert(res >= 0);

        int estimatedSize = NormalizeString(::NormalizationC, _mediaBuf, res, nullptr, 0);
        if (estimatedSize > _maxBytes)
            assert(false); // TODO better
        int actualLength = NormalizeString(::NormalizationC, _mediaBuf, res, _outBuf, _maxBytes);

        if (_outBytesWritten)
            *_outBytesWritten *= actualLength;
    }

    LogHandler createConsoleHandler(LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {
        LogHandler handler{};
        handler.hostAllocate   = _alloc;
        handler.hostDeallocate = _dealloc;
        handler.data           = handler.hostAllocate(sizeof(LoggerData), alignof(LoggerData));
        if (!handler.data)
            ;
        // TOOD
        std::construct_at(std::bit_cast<LoggerData*>(handler.data));

        handler.hostFlush    = []() {};
        handler.hostFilter   = [](void* _data, LogRecord const& record) -> bool { return true; };
        handler.hostCallback = [](void* _data, LogRecord const& record) {
            auto* data = std::bit_cast<LoggerData*>(_data);
            data->writeAsync(record.data, record.numBytes);
        };

        handler.hostCleanup = [](LogHandlerDeallocate _dealloc, void* _data) {
            auto* data = std::bit_cast<LoggerData*>(_data);
            std::destroy_at(data);
            _dealloc(data, sizeof(LoggerData), alignof(LoggerData));
        };

        return handler;
    }
} // namespace dmt