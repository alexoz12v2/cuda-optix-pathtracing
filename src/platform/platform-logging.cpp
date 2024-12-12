module;

#include <array>
#include <string_view>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

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

ConsoleLogger::ConsoleLogger(std::string_view const& prefix, ELogLevel level) :
BaseLogger<ConsoleLogger>(level),
m_prefix(prefix)
{
}

void ConsoleLogger::write(ELogLevel level, std::string_view const& str)
{
    // TODO: os specific IO, filename/line/module/datetime prefix
    if (enabled(level))
    {
        std::printf("%s%s%s\n%s", logcolor::levelToColor(level).data(), m_prefix.data(), str.data(), logcolor::reset.data());
    }
}

void ConsoleLogger::write(ELogLevel level, std::string_view const& str, std::initializer_list<StrBuf> const& list)
{
    // TODO: os specific IO, filename/line/module/datetime prefix
    if (enabled(level))
    {
        m_oss.logInitList(str.data(), list);
        std::printf("%s%s%s\n%s",
                    logcolor::levelToColor(level).data(),
                    m_prefix.data(),
                    m_oss.str().data(),
                    logcolor::reset.data());
    }
}

static_assert(LogDisplay<ConsoleLogger>);

} // namespace dmt