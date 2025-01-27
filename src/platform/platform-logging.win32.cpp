#include "platform-logging.h"

#include "platform-os-utils.h"

#include <charconv>

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

    uint32_t utf16le_From_utf8(char8_t const* DMT_RESTRICT _u8str,
                               uint32_t                    _u8NumBytes,
                               wchar_t* DMT_RESTRICT       _mediaBuf,
                               uint32_t                    _mediaMaxBytes,
                               wchar_t* DMT_RESTRICT       _outBuf,
                               uint32_t                    _maxBytes,
                               uint32_t*                   _outBytesWritten);
    uint32_t appendLocalTime(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes);
    uint32_t appendPhyLoc(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes, LogLocation const& _phyLoc);
    uint32_t appendSrcLoc(wchar_t* DMT_RESTRICT       _buf,
                          uint32_t                    _midBytes,
                          char* DMT_RESTRICT          _midBuf,
                          uint32_t                    _maxBytes,
                          wchar_t* DMT_RESTRICT       _wMidBuf,
                          uint32_t                    _wNumBytes,
                          wchar_t* DMT_RESTRICT       _wNormBuf,
                          uint32_t                    _wNormBytes,
                          std::source_location const& _srcLoc);
    uint32_t appendLogLevelString(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes, ELogLevel level);

    static std::mutex s_consoleLock;

    // https://en.cppreference.com/w/cpp/named_req/BasicLockable
    class Win32Mutex
    {
    public:
        Win32Mutex(wchar_t const* mutexName)
        {
            m_mutex = OpenMutexW(MUTEX_ALL_ACCESS, false, mutexName);
            if (!m_mutex)
                std::abort(); // TOOD better
        }
        Win32Mutex(Win32Mutex const&)                = delete;
        Win32Mutex(Win32Mutex&&) noexcept            = delete;
        Win32Mutex& operator=(Win32Mutex const&)     = delete;
        Win32Mutex& operator=(Win32Mutex&&) noexcept = delete;
        ~Win32Mutex() { CloseHandle(m_mutex); }

        void lock()
        {
            DWORD const waitResult = WaitForSingleObject(m_mutex, INFINITE);
            if (waitResult == WAIT_OBJECT_0)
                return;
            std::abort(); // TODO do something else
        }

        void unlock()
        {
            if (!ReleaseMutex(m_mutex))
                std::abort(); // TODO better
        }

    private:
        HANDLE m_mutex;
    };

    struct SrcLocMemory
    {
        static constexpr uint32_t numBytes  = 256;
        static constexpr uint32_t wNumBytes = numBytes << 1;

        char    midBuf[numBytes]{};
        wchar_t wMidBuf[numBytes]{};
        wchar_t wNormBuf[numBytes]{};
    };

    struct LoggerData
    {
        static WORD constexpr colorMask = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY;
        LoggerData()
        {
            std::unique_ptr<wchar_t[]> name       = std::make_unique<wchar_t[]>(256);
            uint32_t                   nameLength = GetEnvironmentVariableW(L"CHILD_CONOUT", name.get(), 256);
            if (nameLength == 0)
            {
                if (GetLastError() != ERROR_NOT_FOUND)
                    std::abort(); // TODO better
                else
                {
                    hStdOut = CreateFileW(L"CONOUT$",
                                          GENERIC_WRITE,
                                          FILE_SHARE_WRITE,
                                          nullptr,
                                          OPEN_EXISTING,
                                          FILE_FLAG_OVERLAPPED,
                                          nullptr);
                    if (hStdOut == INVALID_HANDLE_VALUE)
                    { // TODO BETTER
                        std::unique_ptr<char[]> buf = std::make_unique<char[]>(2048);
                        uint32_t                len = win32::getLastErrorAsString(buf.get(), 2048);
                        DebugBreak();
                    }
                }
            }
            else
            {
                hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
                if (hStdOut == INVALID_HANDLE_VALUE)
                { // TODO BETTER
                    std::unique_ptr<char[]> buf = std::make_unique<char[]>(2048);
                    uint32_t                len = win32::getLastErrorAsString(buf.get(), 2048);
                    DebugBreak();
                }
            }

            // TODO Check result
            // write at the end ?
            //overlapped.Offset     = 0xFFFF'FFFF;
            //overlapped.OffsetHigh = 0xFFFF'FFFF;
            overlapped.hEvent = CreateEventW(nullptr,  // default security attributes
                                             true,     // manual reset
                                             true,     // start signaled
                                             nullptr); // unnamed event

            // if we are launched from a parent in /CONSOLE:SUBSYSTEM, then it should provide to us a mailbox
            // to signal whenever an IO operation ended
            nameLength = GetEnvironmentVariableW(L"CHILD_MAILBOX", name.get(), 256);
            if (nameLength == 0)
            {
                if (GetLastError() != ERROR_NOT_FOUND)
                    std::abort(); // TODO better
                // leave as INVALID_HANDLE_VALUE hMailbox
            }
            else
            {
                hMailbox = CreateFileW(name.get(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (hMailbox == INVALID_HANDLE_VALUE)
                { // TODO BETTER
                    std::unique_ptr<char[]> buf = std::make_unique<char[]>(2048);
                    uint32_t                len = win32::getLastErrorAsString(buf.get(), 2048);
                    DebugBreak();
                }
            }

            nameLength = GetEnvironmentVariableW(L"CHILD_MUTEX", name.get(), 256);
            if (nameLength == 0)
            {
                if (GetLastError() != ERROR_NOT_FOUND)
                    std::abort(); // TODO better
                // leave as INVALID_HANDLE_VALUE hMailbox
            }
            else
            {
                pMutex = reinterpret_cast<Win32Mutex*>(HeapAlloc(GetProcessHeap(), 0, sizeof(Win32Mutex)));
                if (!pMutex)
                    std::abort(); // TODO better
                std::construct_at(pMutex, name.get());
            }
        }
        LoggerData(LoggerData const&)                = delete;
        LoggerData(LoggerData&&) noexcept            = delete;
        LoggerData& operator=(LoggerData const&)     = delete;
        LoggerData& operator=(LoggerData&&) noexcept = delete;
        ~LoggerData() noexcept
        {
            std::lock_guard lk{s_consoleLock};
            WaitForSingleObject(overlapped.hEvent, INFINITE);
            CloseHandle(overlapped.hEvent);
            std::unique_ptr<wchar_t[]> name       = std::make_unique<wchar_t[]>(256);
            uint32_t                   nameLength = GetEnvironmentVariableW(L"CHILD_CONOUT", name.get(), 256);
            if (nameLength == 0)
            {
                CloseHandle(hStdOut);
            }
            nameLength = GetEnvironmentVariableW(L"CHILD_MUTEX", name.get(), 256);
            if (nameLength != 0)
            {
                std::destroy_at(pMutex);
                HeapFree(GetProcessHeap(), 0, pMutex);
            }
        }

        void waitEmptyMailbox(std::unique_lock<Win32Mutex>& _winLock)
        {
            assert(_winLock.owns_lock());
            DWORD numBytesNextMsg = 0, numMessages = 0;
            if (!GetMailslotInfo(hMailbox, nullptr, &numBytesNextMsg, &numMessages, nullptr))
                std::abort(); // TODO better
            while (numBytesNextMsg != MAILSLOT_NO_MESSAGE)
            {
                _winLock.unlock(); // give parent a chance
                _winLock.lock();
                if (!GetMailslotInfo(hMailbox, nullptr, &numBytesNextMsg, &numMessages, nullptr))
                    std::abort(); // TODO better
            }
        }

        void waitReadyForNext(std::unique_lock<Win32Mutex>& _winLock)
        {
            DWORD query = WaitForSingleObject(overlapped.hEvent, 0);
            if (query == WAIT_TIMEOUT)
                query = WaitForSingleObject(overlapped.hEvent, INFINITE);

            // wait for the mailbox to be empty
            waitEmptyMailbox(_winLock);
        }

        static inline uint32_t printSeparator(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes)
        {
            uint32_t maxChars = _maxBytes >> 1;
            if (maxChars < 3)
                return 0;
            _buf[0] = L' ';
            _buf[1] = L'|';
            _buf[2] = L' ';
            return 3;
        }

        void writeAsync(LogRecord const& record)
        { // if wait failed crash or abandon logging
            std::lock_guard  lk{s_consoleLock};
            std::unique_lock wlk{*pMutex};
            waitReadyForNext(wlk);

            // Set log level for console output
            assert(record.level != ELogLevel::NONE);
            uint8_t const rawLevel = static_cast<std::underlying_type_t<ELogLevel>>(record.level);
            if (!WriteFile(hMailbox, &rawLevel, sizeof(uint8_t), nullptr, nullptr))
                std::abort();

            wchar_t* buf        = normalizedBuffer;
            uint32_t totalBytes = 0;

            // Helper to add content to the buffer
            auto appendToBuffer = [&](auto&& appendFunc, wchar_t* buf, uint32_t& totalBytes) {
                uint32_t bytesAdded = appendFunc(buf, maxBytes - totalBytes);
                buf += bytesAdded;
                totalBytes += bytesAdded << 1;
                assert(maxBytes - totalBytes <= maxBytes);
                return buf;
            };

            // Add components to the buffer
            buf[0] = L'[';
            buf[1] = L'\0';
            totalBytes += 2;
            ++buf;

            // clang-format off
            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return appendLocalTime(buf, maxBytes); }, buf, totalBytes);
            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return printSeparator(buf, maxBytes); }, buf, totalBytes);
            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return appendPhyLoc(buf, maxBytes, record.phyLoc); }, buf, totalBytes);
            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return printSeparator(buf, maxBytes); }, buf, totalBytes);
            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return appendSrcLoc(buf, maxBytes, srcLocMem.midBuf, srcLocMem.numBytes, srcLocMem.wMidBuf, srcLocMem.wNumBytes, srcLocMem.wNormBuf, srcLocMem.wNumBytes, record.srcLoc); }, buf, totalBytes);

            buf[0] = L']';
            buf[1] = L' ';
            totalBytes += 4;
            buf += 2;

            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return appendLogLevelString(buf, maxBytes, record.level); }, buf, totalBytes);
            // clang-format on

            buf[0] = L' ';
            buf[1] = L'-';
            buf[2] = L' ';
            totalBytes += 6;
            buf += 3;

            // Convert and append log record data
            uint32_t bytes = 0;
            utf16le_From_utf8(record.data, record.numBytes, buffer, maxArgChars, buf, maxChars - (totalBytes >> 1), &bytes);
            buf += bytes >> 1;
            totalBytes += bytes;
            buf[0] = L'\0';
            ++buf;
            totalBytes += 2;

            if (!WriteFile(hStdOut, normalizedBuffer, totalBytes, nullptr, &overlapped))
                ; // Handle error
        }

        static constexpr uint32_t maxChars    = 2048;
        static constexpr uint32_t maxArgChars = 1024;
        static constexpr uint32_t maxBytes    = maxChars >> 1;
        OVERLAPPED                overlapped{};
        HANDLE                    hStdOut  = INVALID_HANDLE_VALUE;
        HANDLE                    hMailbox = INVALID_HANDLE_VALUE;
        Win32Mutex*               pMutex   = nullptr;
        SrcLocMemory              srcLocMem;
        wchar_t                   buffer[maxArgChars]{};
        wchar_t                   normalizedBuffer[maxChars]{};
    };

    // TODO move to OS specific utils
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

    uint32_t appendLogLevelString(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes, ELogLevel _level)
    {
        assert(_maxBytes >= 10);
        static wchar_t const levels[toUnderlying(ELogLevel::eCount)][5]{
            {L'T', L'R', L'A', L'C', L'E'},
            {L'L', L'O', L'G', L' ', L' '},
            {L'W', L'A', L'R', L'N', L' '},
            {L'E', L'R', L'R', L'O', L'R'},
        };
        assert(_level != ELogLevel::eCount);
        std::underlying_type_t<ELogLevel> const index = toUnderlying(_level);
        memcpy(_buf, levels[index], sizeof(wchar_t[5]));
        return 5;
    }

    uint32_t appendLocalTime(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes)
    {
        // Get UTC Time
        SYSTEMTIME utcTime{};
        SYSTEMTIME localTime{};
        GetSystemTime(&utcTime);
        if (!SystemTimeToTzSpecificLocalTimeEx(nullptr /*current time zone*/, &utcTime, &localTime))
            std::abort(); // TODO better
        wchar_t* buffer   = _buf;
        uint32_t maxChars = _maxBytes >> 1;
        // YYYY-MM-DD hh:mm:ss
        uint32_t written = _snwprintf(buffer,
                                      maxChars,
                                      L"%04u-%02u-%02u %02u:%02u:%02u",
                                      localTime.wYear,
                                      localTime.wMonth,
                                      localTime.wDay,
                                      localTime.wHour,
                                      localTime.wMinute,
                                      localTime.wSecond);
        return written;
    }

    uint32_t appendSrcLoc(wchar_t* DMT_RESTRICT       _buf,
                          uint32_t                    _maxBytes,
                          char* DMT_RESTRICT          _midBuf,
                          uint32_t                    _midBytes,
                          wchar_t* DMT_RESTRICT       _wMidBuf,
                          uint32_t                    _wNumBytes,
                          wchar_t* DMT_RESTRICT       _wNormBuf,
                          uint32_t                    _wNormBytes,
                          std::source_location const& _srcLoc)
    {
        // filename
        static uint32_t const srcLocFileOffset = strlen(DMT_PROJ_PATH);
        strncpy(_midBuf, _srcLoc.file_name() + srcLocFileOffset, _midBytes);
        _midBuf[_midBytes - 1] = '\0';
        uint32_t len           = strlen(_midBuf);
        uint32_t numChars      = utf16le_From_utf8(std::bit_cast<char8_t const*>(_midBuf),
                                              len,
                                              _wMidBuf,
                                              _wNumBytes,
                                              _wNormBuf,
                                              _wNormBytes,
                                              nullptr);
        _wNormBuf[numChars]    = L'\0';
        uint32_t maxChars      = _maxBytes >> 1;
        uint32_t written       = _snwprintf(_buf, maxChars, L"%s:", _wNormBuf);
        maxChars -= written;
        _buf += written;
        if (maxChars == 0)
            return written;

        // function name
        strncpy(_midBuf, _srcLoc.function_name(), _midBytes);
        _midBuf[_midBytes - 1] = '\0';
        len                    = strlen(_midBuf);
        numChars = utf16le_From_utf8(std::bit_cast<char8_t const*>(_midBuf), len, _wMidBuf, _wNumBytes, _wNormBuf, _wNormBytes, nullptr);
        _wNormBuf[numChars]     = L'\0';
        uint32_t const written1 = _snwprintf(_buf, maxChars, L"%s:", _wNormBuf);
        written += written1;
        maxChars -= written1;
        _buf += written1;
        if (maxChars == 0)
            return written;

        // line number
        written += _snwprintf(_buf, maxChars, L"%u", _srcLoc.line());
        return written;
    }

    uint32_t appendPhyLoc(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes, LogLocation const& _phyLoc)
    {
        uint32_t written  = 0;
        uint32_t maxChars = _maxBytes >> 1;
        if (_phyLoc.where == LogLocation::hostNum)
        {
            written = _snwprintf(_buf, maxChars, L"CPU, pid %5llX, tid %5llX", _phyLoc.loc.host.pid, _phyLoc.loc.host.tid);
        }
        else
        {
            // GPU location branch
            auto const& dev = _phyLoc.loc.dev;

            // clang-format off
            written = _snwprintf(_buf, maxChars,
                                 L"GPU, grid (block: [%d, %d, %d], thread: [%d, %d, %d]), lane: %d, warp: %d",
                                 dev.blockX, dev.blockY, dev.blockZ, // Block coordinates
                                 dev.threadX, dev.threadY, dev.threadZ, // Thread coordinates
                                 dev.lane, dev.warp);   // lane and Warp
            // clang-format on
        }
        return written;
    }

    bool createConsoleHandler(LogHandler& _out, LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {
        _out.hostAllocate   = _alloc;
        _out.hostDeallocate = _dealloc;
        _out.data           = _out.hostAllocate(sizeof(LoggerData), alignof(LoggerData));
        if (!_out.data)
            return false;
        std::construct_at(std::bit_cast<LoggerData*>(_out.data));

        _out.hostFlush = [](void* _data) {
            auto*            data = std::bit_cast<LoggerData*>(_data);
            std::unique_lock wlk{*data->pMutex};
            data->waitReadyForNext(wlk);
        };
        _out.hostFilter   = [](void* _data, LogRecord const& record) -> bool { return true; };
        _out.hostCallback = [](void* _data, LogRecord const& record) {
            auto* data = std::bit_cast<LoggerData*>(_data);
            data->writeAsync(record);
        };

        _out.hostCleanup = [](LogHandlerDeallocate _dealloc, void* _data) {
            auto* data = std::bit_cast<LoggerData*>(_data);
            std::destroy_at(data);
            _dealloc(data, sizeof(LoggerData), alignof(LoggerData));
        };

        return true;
    }
} // namespace dmt