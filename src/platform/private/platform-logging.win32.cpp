#include "platform-logging.h"

#include "platform-os-utils.win32.h"

#include <charconv>

#include <windows.h> // should be included exactly once in every translation unit in which you need definitions

namespace dmt {
    static uint32_t appendLocalTime(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes);
    static uint32_t appendPhyLoc(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes, LogLocation const& _phyLoc);
    static uint32_t appendSrcLoc(
        wchar_t* DMT_RESTRICT       _buf,
        uint32_t                    _midBytes,
        char* DMT_RESTRICT          _midBuf,
        uint32_t                    _maxBytes,
        wchar_t* DMT_RESTRICT       _wMidBuf,
        uint32_t                    _wNumBytes,
        wchar_t* DMT_RESTRICT       _wNormBuf,
        uint32_t                    _wNormBytes,
        std::source_location const& _srcLoc);
    static uint32_t appendLogLevelString(wchar_t* DMT_RESTRICT _buf, uint32_t _maxBytes, ELogLevel level);

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
                if (auto err = GetLastError(); err != ERROR_ENVVAR_NOT_FOUND)
                    std::abort(); // TODO better
                else
                    return; // leave hStdOut to `INVALID_HANDLE_VALUE` to signal that console logging won't be performed
            }
            else
            {
                hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
                if (hStdOut == INVALID_HANDLE_VALUE)
                { // TODO BETTER
                    std::unique_ptr<char[]> buf = std::make_unique<char[]>(2048);
                    uint32_t                len = ::dmt::os::win32::getLastErrorAsString(buf.get(), 2048);
                    DebugBreak();
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
                    std::abort(); // TODO better

                hMailbox = CreateFileW(name.get(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (hMailbox == INVALID_HANDLE_VALUE)
                { // TODO BETTER
                    std::unique_ptr<char[]> buf = std::make_unique<char[]>(2048);
                    uint32_t                len = ::dmt::os::win32::getLastErrorAsString(buf.get(), 2048);
                    DebugBreak();
                }

                nameLength = GetEnvironmentVariableW(L"CHILD_MUTEX", name.get(), 256);
                if (nameLength == 0)
                    std::abort(); // TODO better

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
            if (hStdOut != INVALID_HANDLE_VALUE)
            {
                WaitForSingleObject(overlapped.hEvent, INFINITE);
                CloseHandle(overlapped.hEvent);
                CloseHandle(hStdOut);
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
            if (maxChars + 1 < 3)
                return 0;
            _buf[0] = L' ';
            _buf[1] = L'|';
            _buf[2] = L' ';
            _buf[3] = L'\0';
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
            buf[2] = L'\0';
            totalBytes += 4;
            buf += 2;

            buf = appendToBuffer([&](wchar_t* buf, uint32_t maxBytes) { return appendLogLevelString(buf, maxBytes, record.level); }, buf, totalBytes);
            // clang-format on

            buf[0] = L' ';
            buf[1] = L'-';
            buf[2] = L' ';
            buf[3] = L'\0';
            totalBytes += 6;
            buf += 3;

            // Convert and append log record data
            uint32_t bytes = 0;
            ::dmt::os::win32::utf16le_From_utf8(record.data,
                                                record.numBytes,
                                                buffer,
                                                maxArgChars,
                                                buf,
                                                maxChars - (totalBytes >> 1),
                                                &bytes);
            buf += bytes >> 1;
            totalBytes += bytes;
            buf[0] = L'\0';
            ++buf;
            totalBytes += 2;

            if (!WriteFile(hStdOut, normalizedBuffer, totalBytes, nullptr, &overlapped))
                ; // Handle error
        }

        static constexpr uint32_t maxChars    = 4096;
        static constexpr uint32_t maxArgChars = 2048;
        static constexpr uint32_t maxBytes    = maxChars >> 1;
        OVERLAPPED                overlapped{};
        HANDLE                    hStdOut  = INVALID_HANDLE_VALUE;
        HANDLE                    hMailbox = INVALID_HANDLE_VALUE;
        Win32Mutex*               pMutex   = nullptr;
        SrcLocMemory              srcLocMem;
        wchar_t                   buffer[maxArgChars]{};
        wchar_t                   normalizedBuffer[maxChars]{};
    };

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
#define STRING(...) #__VA_ARGS__
        // filename
        static uint32_t const srcLocFileOffset = strlen(STRING(DMT_PROJ_PATH));
        strncpy(_midBuf, _srcLoc.file_name() + srcLocFileOffset, _midBytes);
        _midBuf[_midBytes - 1]   = '\0';
        uint32_t len             = strlen(_midBuf);
        uint32_t numChars        = ::dmt::os::win32::utf16le_From_utf8(std::bit_cast<char const*>(_midBuf),
                                                                len,
                                                                _wMidBuf,
                                                                _wNumBytes,
                                                                _wNormBuf,
                                                                _wNormBytes,
                                                                nullptr);
        _wNormBuf[numChars >> 1] = L'\0';
        uint32_t maxChars        = _maxBytes >> 1;
        uint32_t written         = _snwprintf(_buf, maxChars, L"%s:", _wNormBuf);
        maxChars -= written;
        _buf += written;
        if (maxChars == 0)
            return written;

        // function name
        strncpy(_midBuf, _srcLoc.function_name(), _midBytes);
        _midBuf[_midBytes - 1]   = '\0';
        len                      = strlen(_midBuf);
        numChars                 = ::dmt::os::win32::utf16le_From_utf8(std::bit_cast<char const*>(_midBuf),
                                                       len,
                                                       _wMidBuf,
                                                       _wNumBytes,
                                                       _wNormBuf,
                                                       _wNormBytes,
                                                       nullptr);
        _wNormBuf[numChars >> 1] = L'\0';
        uint32_t const written1  = _snwprintf(_buf, maxChars, L"%s:", _wNormBuf);
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


    static int getExeSubsystem()
    {
        HMODULE hModule = GetModuleHandleW(NULL); // base of main EXE
        if (!hModule)
            return -1;

        BYTE* base = (BYTE*)hModule;

        IMAGE_DOS_HEADER* dosHeader = (IMAGE_DOS_HEADER*)base;
        if (dosHeader->e_magic != IMAGE_DOS_SIGNATURE)
            return -1;

        IMAGE_NT_HEADERS* ntHeader = (IMAGE_NT_HEADERS*)(base + dosHeader->e_lfanew);
        if (ntHeader->Signature != IMAGE_NT_SIGNATURE)
            return -1;

        return ntHeader->OptionalHeader.Subsystem;
    }

    inline constexpr WORD white  = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
    inline constexpr WORD red    = FOREGROUND_RED | FOREGROUND_INTENSITY;
    inline constexpr WORD yellow = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
    inline constexpr WORD olive  = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY;

    static void setColor(WORD c)
    {
        static constexpr WORD consoleColorMask = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
        CONSOLE_SCREEN_BUFFER_INFO consoleInfo{};
        GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &consoleInfo);
        WORD attribute = (consoleInfo.wAttributes & ~consoleColorMask) | c;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), attribute);
    }

    bool createConsoleHandler(LogHandler& _out, LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {

        _out.hostAllocate   = _alloc;
        _out.hostDeallocate = _dealloc;

        int const subsystem = getExeSubsystem();

        if (subsystem == IMAGE_SUBSYSTEM_WINDOWS_CUI)
        {
            SetConsoleCP(CP_UTF8);
            SetConsoleOutputCP(CP_UTF8);

            _out.data = _out.hostAllocate(sizeof(SpinLock) + 8192, alignof(SpinLock));
            if (!_out.data)
                return false;
            std::construct_at(reinterpret_cast<SpinLock*>(_out.data));
            _out.hostFlush    = [](void* _data) {};
            _out.hostFilter   = [](void* _data, LogRecord const& record) -> bool { return true; };
            _out.hostCallback = [](void* _data, LogRecord const& record) {
                auto*                               data = std::bit_cast<SpinLock*>(_data);
                std::lock_guard                     lk{*data};
                auto*                               buffer = reinterpret_cast<unsigned char*>(data + 1);
                std::pmr::monotonic_buffer_resource mem{buffer, 8192, std::pmr::null_memory_resource()};
                if (record.level == ELogLevel::eLog)
                    setColor(white);
                else if (record.level == ELogLevel::eWarn)
                    setColor(yellow);
                else if (record.level == ELogLevel::eTrace)
                    setColor(olive);
                else
                    setColor(red);

                std::pmr::wstring wData = os::win32::utf16FromUtf8({record.data, record.numBytes}, &mem);
                WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE), wData.data(), static_cast<DWORD>(wData.size()), nullptr, nullptr);

                setColor(white);
            };
            _out.hostCleanup = [](LogHandlerDeallocate _dealloc, void* _data) {
                auto*           data = std::bit_cast<SpinLock*>(_data);
                std::lock_guard lk{*data};
                std::destroy_at(data);
                _dealloc(data, sizeof(SpinLock) + 8192, alignof(SpinLock));
            };
        }
        else if (subsystem == IMAGE_SUBSYSTEM_WINDOWS_GUI)
        {
            _out.data = _out.hostAllocate(sizeof(LoggerData), alignof(LoggerData));
            if (!_out.data)
                return false;
            std::construct_at(std::bit_cast<LoggerData*>(_out.data));

            _out.hostFlush = [](void* _data) {
                auto* data = std::bit_cast<LoggerData*>(_data);
                if (data->hStdOut != INVALID_HANDLE_VALUE)
                {
                    std::unique_lock wlk{*data->pMutex};
                    data->waitReadyForNext(wlk);
                }
            };
            _out.hostFilter   = [](void* _data, LogRecord const& record) -> bool { return true; };
            _out.hostCallback = [](void* _data, LogRecord const& record) {
                auto* data = std::bit_cast<LoggerData*>(_data);
                if (data->hStdOut != INVALID_HANDLE_VALUE)
                {
                    data->writeAsync(record);
                }
            };

            _out.hostCleanup = [](LogHandlerDeallocate _dealloc, void* _data) {
                auto* data = std::bit_cast<LoggerData*>(_data);
                std::destroy_at(data);
                _dealloc(data, sizeof(LoggerData), alignof(LoggerData));
            };
        }
        else
        {
            return false;
        }

        return true;
    }
} // namespace dmt