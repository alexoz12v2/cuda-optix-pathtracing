module;

#include <array>
#include <chrono>
#include <memory>
#include <source_location>
#include <string_view>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

// TODO refactor in IO utilities
#if defined(DMT_OS_LINUX)
#include <signal.h>
#include <sys/ioctl.h>
#include <unistd.h>
#if defined(_POSIX_ASYNCHRONOUS_IO)
#include <aio.h> // https://man7.org/linux/man-pages/man7/aio.7.html https://www.gnu.org/software/libc/manual/html_node/Asynchronous-I_002fO.html
#include <fcntl.h>
#endif
#elif defined(DMT_OS_WINDOWS)
#include <windows.h> // should be included exactly once in every translation unit in which you need definitions
#endif

// https://github.com/fmtlib/fmt/blob/master/include/fmt/format.h, line 4153

module platform;

namespace dmt
{

CircularOStringStream::CircularOStringStream()
{
    std::memset(m_buffer, 0, bufferSize);
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
        assert(false); // you shouldn't get here
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
    m_pos = 0;
    std::memset(m_buffer, 0, bufferSize);
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

// TODO refactor in a module handling platform OS issues
#if defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO) // <- required macro
static_assert(sizeof(aiocb) <= 168);
struct alignas(256) PaddedAioCb
{
    aiocb acb;
};
static_assert(sizeof(PaddedAioCb) == 256 && std::is_trivial_v<PaddedAioCb> && std::is_standard_layout_v<PaddedAioCb>);

// TOOD add threaded IO management when thread IO branch is ready
// TODO add proper memory management
class LinuxAsyncIOManager
{
public:
    static inline constexpr uint32_t numAios     = 4;
    static inline constexpr uint32_t lineSize    = 2047;
    static inline constexpr uint32_t maxAttempts = 10;
    LinuxAsyncIOManager();
    LinuxAsyncIOManager(LinuxAsyncIOManager const&)            = delete;
    LinuxAsyncIOManager(LinuxAsyncIOManager&&)                 = delete;
    LinuxAsyncIOManager& operator=(LinuxAsyncIOManager const&) = delete;
    LinuxAsyncIOManager& operator=(LinuxAsyncIOManager&&)      = delete;
    ~LinuxAsyncIOManager();

    // Enqueue IO work to either STDOUT or STDERR
    // teh work should NOT have the
    bool     enqueue(int32_t idx, size_t size);
    uint32_t findFirstFreeBlocking(int32_t fildes);
    char*    operator[](uint32_t idx);

    // Poll for completion of IO operations
    void sync();

private:
    struct Line
    {
        char buf[lineSize];
        bool free;
    };
    // Helper method to initialize the AIO control blocks
    void initAio();

    // Synchronize writes by flushing the current file descriptor
    void syncIfNeeded(int newFd);


    bool handleStatus(int32_t status, aiocb& outAio, Line& outLine);

    PaddedAioCb*  m_aioQueue;
    Line*         m_lines;
    int32_t       m_currentFd = STDOUT_FILENO;
    unsigned char padding[40];
};

static_assert(std::is_standard_layout_v<LinuxAsyncIOManager> &&
              sizeof(LinuxAsyncIOManager) == ConsoleLogger::asyncIOClassSize);

LinuxAsyncIOManager::LinuxAsyncIOManager() :
m_aioQueue(reinterpret_cast<PaddedAioCb*>(std::aligned_alloc(alignof(PaddedAioCb), numAios * sizeof(PaddedAioCb)))),
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
        std::memset(&m_aioQueue[i], 0, sizeof(PaddedAioCb));
        m_aioQueue[i].acb.aio_fildes     = m_currentFd;
        m_aioQueue[i].acb.aio_offset     = 0; // Default to write operation
        m_aioQueue[i].acb.aio_buf        = reinterpret_cast<volatile void*>(m_lines[i].buf);
        m_aioQueue[i].acb.aio_lio_opcode = LIO_WRITE; // Default to write operation
        // m_aioQueue[i].aio_reqprio    =, scheduling priority. Requires additional macros
        m_aioQueue[i].acb.aio_sigevent.sigev_notify = SIGEV_NONE; // SIGEV_NONE, SIGEV_SIGNAL -> sigev_signo, SIGEV_THREAD -> sigev_notify_attributes, specify thread: (linux only) sigev_notify_thread_id (used only by timers)
        m_aioQueue[i].acb.aio_lio_opcode = LIO_WRITE; // only used by lio_listio when you schedule multiple operations
        m_lines[i].free                  = true;
    }
}

LinuxAsyncIOManager::~LinuxAsyncIOManager()
{
    sync();
    std::free(m_aioQueue);
    std::free(m_lines);
}

char* LinuxAsyncIOManager::operator[](uint32_t idx)
{
    return m_lines[idx].buf;
}

uint32_t LinuxAsyncIOManager::findFirstFreeBlocking(int32_t fildes)
{
    // Ensure we complete previous operations before starting a new one
    // necessary only if we are switching from STDOUT to STDERR
    syncIfNeeded(fildes);

    // Find an available aiocb slot (simple round-robin or any available slot)
    while (true) // if everything is full, then poll
    {
        for (uint32_t i = 0; i < numAios; ++i)
        {
            if (aio_error(&m_aioQueue[i].acb) != EINPROGRESS)
            {
                m_lines[idx].free              = false; // TODO remove free
                m_aioQueue[i].acb.aio_fildes = fildes;
                return i;
            }
        }
    }
}

void LinuxAsyncIOManager::syncIfNeeded(int newFd)
{
    // Only synchronize if we are switching file descriptors
    if (newFd != m_currentFd)
    {
        sync();

        // Once completed, switch the file descriptor
        m_currentFd = newFd;

        initAio();
    }
}

bool LinuxAsyncIOManager::enqueue(int32_t idx, size_t size)
{
    // no sync needed as handled by findFirstFreeBlocking
    // the m_lines[idx].buf should be written externally
    // Find an available aiocb slot (simple round-robin or any available slot)
    assert(aio_error(&m_aioQueue[idx].acb) != EINPROGRESS && m_lines[idx].free);
    m_aioQueue[idx].acb.aio_nbytes = size;
    int status                     = aio_write(&m_aioQueue[idx].acb);
    return handleStatus(status, m_aioQueue[idx].acb, m_lines[idx]);
}

bool LinuxAsyncIOManager::handleStatus(int32_t status, aiocb& outAio, Line& outLine)
{

    assert(status != EBADF && "Asynchronous write File Descriptor invalid!");
    assert(status != EINVAL && "Invalid `aio_offset` or `aio_reqprio`!");
    uint32_t attempt = 1;
    for (; status == EAGAIN && attempt != maxAttempts; ++attempt)
    {
        status = aio_write(&outAio);
    }

    if (attempt != maxAttempts)
    {
        outLine.free = false;
        return false;
    }

    return true;
}

void LinuxAsyncIOManager::sync()
{
    aiocb syncAio      = {};
    syncAio.aio_fildes = m_currentFd;
    auto**  p          = reinterpret_cast<aiocb**>(alloca((LinuxAsyncIOManager::numAios + 1) * sizeof(aiocb const*)));
    int32_t current    = 0;
    for (int32_t i = 0; i != LinuxAsyncIOManager::numAios; ++i)
    {
        if (aio_error(&m_aioQueue[i].acb) == EINPROGRESS)
        {
            p[current++] = &m_aioQueue[i].acb;
        }
    }

    // Flush the current file descriptor. Schedule an async flush operation...
    int32_t  out     = EAGAIN;
    uint32_t attempt = 0;
    while (attempt++ != maxAttempts && out == EAGAIN)
    {
        out = aio_fsync(O_SYNC, &syncAio);
        assert(out != EBADF && "Invalid File Descriptor");
        assert(out != EINVAL && "Invalid operation `O_SYNC` or synchonization not supported");
        assert(out != ENOSYS && "Async IO not supported, you shouldn't call me");
    }
    p[current++] = &syncAio;

    // ... Then wait for the flush and all operation which came before it
    // for some reason, it doesn't work when both STDOUT_FILENO and STDERR_FILENO are involved
    // so we are keeping it with only STDOUT_FILENO
    aio_suspend(p, current, nullptr); // returns when at least one is finished
}


// standard layout to be castable and memcopied from array of bytes + size requirement
// TODO: Extract static circular queue
#elif defined(DMT_OS_WINDOWS)

// TODO extract string buffer allocation to a base class
class WindowsAsyncIOManager
{
public:
    static inline constexpr uint32_t numAios     = 4;
    static_assert(numAios < 5);
    static inline constexpr uint32_t lineSize    = 2048;
    static inline constexpr uint32_t maxAttempts = 10;

    WindowsAsyncIOManager();
    WindowsAsyncIOManager(WindowsAsyncIOManager const&)       = delete;
    WindowsAsyncIOManager(WindowsAsyncIOManager&&)            = delete;
    WindowsAsyncIOManager& operator=(WindowsAsyncIOManager const&) = delete;
    WindowsAsyncIOManager& operator=(WindowsAsyncIOManager&&) = delete;
    ~WindowsAsyncIOManager();

    uint32_t findFirstFreeBlocking();
    bool enqueue(int32_t idx, size_t size);
    char*    operator[](uint32_t idx);

private:
    void sync();
    void initAio();
    int32_t waitForEvents(DWORD timeout, bool waitAll);

	struct Line {
        char buf[lineSize];
    };

    struct OverlappedWrite {
        OVERLAPPED overlapped;
    };

    HANDLE m_hStdOut = nullptr;
    HANDLE           m_hBuffer[numAios]{};
	OverlappedWrite* m_aioQueue;
    Line* m_lines;
    unsigned char m_padding[8];
};

static_assert(std::is_standard_layout_v<WindowsAsyncIOManager> 
    && sizeof(WindowsAsyncIOManager) == ConsoleLogger::asyncIOClassSize);

WindowsAsyncIOManager::WindowsAsyncIOManager() : 
m_hStdOut(GetStdHandle(STD_OUTPUT_HANDLE)),
m_aioQueue(reinterpret_cast<OverlappedWrite*>(std::malloc(numAios * sizeof(OverlappedWrite)))),
m_lines(reinterpret_cast<Line *>(std::malloc(numAios * sizeof(Line))))
{
    if (!m_aioQueue || !m_lines)
        std::abort();

	initAio();
}

void WindowsAsyncIOManager::initAio() {
    SECURITY_ATTRIBUTES secAttrs{
        .nLength              = sizeof(SECURITY_ATTRIBUTES),
        .lpSecurityDescriptor = nullptr, // you have the same rights/permissions as the user runnning the process
        .bInheritHandle       = true, // inherited by child processes
    };

    for (uint32_t i = 0; i != numAios; ++i) {
        m_aioQueue[i].overlapped.hEvent = CreateEventA(&secAttrs, /*manualReset*/ true, /*startSignaled*/ true, /*name*/ nullptr);
        assert(m_aioQueue[i].overlapped.hEvent != nullptr && "Couldn't create event!");
    }
}

char* WindowsAsyncIOManager::operator[](uint32_t idx) {
    return m_lines[idx].buf;
}

uint32_t WindowsAsyncIOManager::findFirstFreeBlocking()
{
    // Try to find a free event with polling (timeout = 0)
    int32_t freeIndex = waitForEvents(0, /*waitAll=*/false);
    if (freeIndex >= 0) {
        // io operation resets the event specified by the hEvent member of the OVERLAPPED structure to a 
        // nonsignaled state when it begins the I/O operation. Therefore, the caller does not need to do that.
        // ResetEvent(m_aioQueue[freeIndex].overlapped.hEvent); 
        return static_cast<uint32_t>(freeIndex); // Return immediately if a free event is found
    }

    // If no free event is found, block until one is available
    // https://learn.microsoft.com/en-gb/windows/win32/api/fileapi/nf-fileapi-createfilea?redirectedfrom=MSDN
    // ^ windows actually doesn't support overlapped output to console. Therefore, if no one is free,
    // reset all events and pick the first one, hoping everything is fine :)
    for (uint32_t i = 0; i != numAios; ++i) {
        ResetEvent(m_aioQueue[i].overlapped.hEvent);
    }

    return 0;
}

void WindowsAsyncIOManager::sync()
{
    // overlapped not supported,
    // waitForEvents(INFINITE, /*waitForAll*/ true);
}

int32_t WindowsAsyncIOManager::waitForEvents(DWORD timeout, bool waitAll) {
	// https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects
    // try to find a free one with polling, ie WaitForSingleObject with timeout = 0
    // to use wait for multiple objects, we need all event objects in a single buffer
    // Prepare the buffer with event handles
    for (uint32_t i = 0; i < numAios; ++i) {
        std::memcpy(&m_hBuffer[i], &m_aioQueue[i].overlapped.hEvent, sizeof(HANDLE));
    }

    // Wait for events
    DWORD ret = WaitForMultipleObjects(numAios, m_hBuffer, waitAll, timeout);

    // Check for errors or abandoned events
    assert((ret < WAIT_ABANDONED_0 || ret > WAIT_ABANDONED_0 + numAios - 1) && "I/O operation aborted? What?");
    assert(ret != WAIT_FAILED && "Wait operation failed. Call GetLastError().");

    if (WAIT_OBJECT_0 <= ret && ret <= WAIT_OBJECT_0 + numAios - 1) {
        return ret - WAIT_OBJECT_0; // Return the index of the signaled event
    }

    if (ret == WAIT_TIMEOUT) {
        return -1;
    }

    assert(false && "Unexpected state reached in waitForEvents.");
    return -2; // Unreachable, satisfies compiler
}

WindowsAsyncIOManager::~WindowsAsyncIOManager() {
    sync();

    for (uint32_t i = 0; i != numAios; ++i) {
        CloseHandle(m_aioQueue[i].overlapped.hEvent);
    }
    std::free(m_aioQueue);
    std::free(m_lines);
}

bool WindowsAsyncIOManager::enqueue(int32_t idx, size_t size) {
    bool started = WriteFile(
        m_hStdOut, 
        m_lines[idx].buf,
        size,
        nullptr, 
        &m_aioQueue[idx].overlapped
    );

    if (started && GetLastError() == ERROR_IO_PENDING) {
        // io started
        return false;
    } else {
        return true;
    }
}

#endif

#define DMT_ENABLE_ASYNC_LOG 1

ConsoleLogger::ConsoleLogger(ELogLevel level) : BaseLogger<ConsoleLogger>(level)
{
#if DMT_ENABLE_ASYNC_LOG && defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO) // <- required macro
    assert(reinterpret_cast<std::uintptr_t>(m_asyncIOClass) % alignof(LinuxAsyncIOManager) == 0);
    std::construct_at<LinuxAsyncIOManager>(reinterpret_cast<LinuxAsyncIOManager*>(m_asyncIOClass));
#elif DMT_ENABLE_ASYNC_LOG && defined(DMT_OS_WINDOWS)
    assert(reinterpret_cast<std::uintptr_t>(m_asyncIOClass) % alignof(WindowsAsyncIOManager) == 0);
    std::construct_at<WindowsAsyncIOManager>(reinterpret_cast<WindowsAsyncIOManager*>(m_asyncIOClass));
#endif
}

ConsoleLogger::~ConsoleLogger()
{
#if DMT_ENABLE_ASYNC_LOG && defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO) // <- required macro
    std::destroy_at<LinuxAsyncIOManager>(reinterpret_cast<LinuxAsyncIOManager*>(m_asyncIOClass));
#elif DMT_ENABLE_ASYNC_LOG && defined(DMT_OS_WINDOWS)
    std::destroy_at<WindowsAsyncIOManager>(reinterpret_cast<WindowsAsyncIOManager*>(m_asyncIOClass));
#endif
}


void ConsoleLogger::write(ELogLevel level, std::string_view const& str, std::source_location const loc)
{
    if (enabled(level))
    {
        std::string_view date      = getCurrentTimestamp();
        std::string_view file_name = getRelativeFileName(loc.file_name());
#if DMT_ENABLE_ASYNC_LOG
        logMessageAsync(level, date, file_name, loc.function_name(), loc.line(), stringFromLevel(level), str);
#else
        logMessage(level, date, file_name, loc.function_name(), loc.line(), stringFromLevel(level), str);
#endif
    }
}

void ConsoleLogger::write(ELogLevel                            level,
                          std::string_view const&              str,
                          std::initializer_list<StrBuf> const& list,
                          std::source_location const           loc)
{
    if (enabled(level))
    {
        std::string_view date      = getCurrentTimestamp();
        std::string_view file_name = getRelativeFileName(loc.file_name());

        m_oss.logInitList(str.data(), list);
#if DMT_ENABLE_ASYNC_LOG
        logMessageAsync(level, date, file_name, loc.function_name(), loc.line(), stringFromLevel(level), m_oss.str());
#else
        logMessage(level, date, file_name, loc.function_name(), loc.line(), stringFromLevel(level), m_oss.str());
#endif
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

static uint32_t logToBuffer(
    char*                   buffer,
    uint32_t                size,
    ELogLevel               level,
    std::string_view const& date,
    std::string_view const& fileName,
    std::string_view const& functionName,
    uint32_t                line,
    std::string_view const& levelStr,
    std::string_view const& content)
{
    int32_t status = std::snprintf(buffer,
                                   size,
                                   "%s[%s %s:%s:%u] %s <> %s\n%s",
                                   logcolor::colorFromLevel(level).data(),
                                   date.data(),
                                   fileName.data(),
                                   functionName.data(),
                                   line,
                                   levelStr.data(),
                                   content.data(),
                                   logcolor::reset.data());
    assert(status > 0 && "could not log to buffer");
    return static_cast<uint32_t>(status);
}

// TODO: you can even wait for a given milliseconds amoount before scheduling the IO requests
void ConsoleLogger::logMessageAsync(
    ELogLevel               level,
    std::string_view const& date,
    std::string_view const& fileName,
    std::string_view const& functionName,
    uint32_t                line,
    std::string_view const& levelStr,
    std::string_view const& content)
{
#if defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO) // <- required macro
    // recover class
    LinuxAsyncIOManager& clazz   = *reinterpret_cast<LinuxAsyncIOManager*>(&m_asyncIOClass);
    uint32_t             freeIdx = clazz.findFirstFreeBlocking(STDOUT_FILENO);
    uint32_t sz = logToBuffer(clazz[freeIdx], LinuxAsyncIOManager::lineSize, level, date, fileName, functionName, line, levelStr, content);
    if (clazz.enqueue(freeIdx, sz))
    {
        logMessage(level, date, fileName, functionName, line, levelStr, content);
    }
#elif defined(DMT_OS_WINDOWS)
    WindowsAsyncIOManager& clazz = *reinterpret_cast<WindowsAsyncIOManager*>(&m_asyncIOClass);
    uint32_t               freeIdx = clazz.findFirstFreeBlocking();
    uint32_t               sz      = logToBuffer(clazz[freeIdx], WindowsAsyncIOManager::lineSize, level, date, fileName, functionName, line, levelStr, content);
    if (clazz.enqueue(freeIdx, sz)) {
		logMessage(level, date, fileName, functionName, line, levelStr, content);
    }
#else
    logMessage(level, date, fileName, functionName, line, levelStr, content);
#endif
}

// Helper function to get a relative file name
std::string_view ConsoleLogger::getRelativeFileName(std::string_view fullPath)
{
    static constexpr std::string_view projPath = DMT_PROJ_PATH;
    assert(fullPath.starts_with(projPath) && "the specified file is outside of the project!");
    fullPath.remove_prefix(projPath.size() + 1);
    return fullPath;
}

// Helper function to get the current timestamp
std::string_view ConsoleLogger::getCurrentTimestamp()
{
    static thread_local char buf[timestampMax];
    std::time_t              now     = std::time(nullptr);
    std::tm                  tstruct = *std::localtime(&now);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}

std::string_view ConsoleLogger::cwd()
{
    static thread_local char buf[pathMax];
#if defined(DMT_OS_LINUX)
    char const* ptr = ::getcwd(buf, pathMax);
    return ptr;
#elif defined(DMT_OS_WINDOWS)
    DWORD status = GetCurrentDirectory(pathMax, buf);
    assert(status > 0 && "Could not get the current working directory");
    return {buf, status};
#else
#error "Platform not expected"
#endif
}

static_assert(LogDisplay<ConsoleLogger>);

} // namespace dmt