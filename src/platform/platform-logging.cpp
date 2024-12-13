module;

#include <array>
#include <chrono>
#include <source_location>
#include <string_view>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#if defined(DMT_OS_LINUX)
#include <unistd.h>
#elif defined(DMT_OS_WINDOWS)
#include <winbase.h> // https://learn.microsoft.com/en-us/windows/win32/api/winbase/
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

ConsoleLogger::ConsoleLogger(std::string_view const& prefix, ELogLevel level) : BaseLogger<ConsoleLogger>(level)
{
}

void ConsoleLogger::write(ELogLevel level, std::string_view const& str, std::source_location const loc)
{
    if (enabled(level))
    {
        std::string_view date      = getCurrentTimestamp();
        std::string_view file_name = getRelativeFileName(loc.file_name());
        logMessage(level, date, file_name, loc.function_name(), loc.line(), stringFromLevel(level), str);
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
        logMessage(level, date, file_name, loc.function_name(), loc.line(), stringFromLevel(level), m_oss.str());
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

// Helper function to get a relative file name
std::string_view ConsoleLogger::getRelativeFileName(std::string_view fullPath)
{
    static constexpr std::string_view projPath = DMT_PROJ_PATH;
    if (fullPath.starts_with(projPath))
    {
        fullPath.remove_prefix(projPath.size());
        if (fullPath.starts_with('/'))
        {
            fullPath.remove_prefix(1); // Remove leading slash
        }
    }
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