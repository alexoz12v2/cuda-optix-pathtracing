#include "platform-logging.h"

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

    ConsoleLogger::~ConsoleLogger() { m_IOClassInterface.destructor(m_asyncIOClass); }

    void ConsoleLogger::write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
    {
        if (level >= this->level)
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
        if (level >= this->level)
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
        bool res = m_IOClassInterface.tryAsyncLog(m_asyncIOClass, level, date, fileName, functionName, line, levelStr, content);
        if (res)
        {
            logMessage(level, date, fileName, functionName, line, levelStr, content);
        }
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

    ConsoleLogger::ConsoleLogger(ConsoleLogger&& other) : level(other.level)
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

    size_t ConsoleLogger::maxLogArgBytes() const
    {
        return std::min(static_cast<size_t>(BaseAsyncIOManager::lineSize) >> 1, m_oss.maxLogArgBytes());
    }

    // LoggingContext -------------------------------------------------------------------------------------------------
    LoggingContext::LoggingContext() :
    logger(ConsoleLogger::create()),
    start(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
              .count())
    {
    }

    void LoggingContext::log(std::string_view const& str, std::source_location const& loc)
    {
        logger.write(ELogLevel::LOG, str, loc);
    }

    void LoggingContext::log(std::string_view const&              str,
                             std::initializer_list<StrBuf> const& list,
                             std::source_location const&          loc)
    {
        logger.write(ELogLevel::LOG, str, list, loc);
    }

    void LoggingContext::warn(std::string_view const& str, std::source_location const& loc)
    {
        logger.write(ELogLevel::WARNING, str, loc);
    }

    void LoggingContext::warn(std::string_view const&              str,
                              std::initializer_list<StrBuf> const& list,
                              std::source_location const&          loc)
    {
        logger.write(ELogLevel::WARNING, str, list, loc);
    }

    void LoggingContext::error(std::string_view const& str, std::source_location const& loc)
    {
        logger.write(ELogLevel::ERR, str, loc);
    }

    void LoggingContext::error(std::string_view const&              str,
                               std::initializer_list<StrBuf> const& list,
                               std::source_location const&          loc)
    {
        logger.write(ELogLevel::ERR, str, list, loc);
    }

    void LoggingContext::trace(std::string_view const& str, std::source_location const& loc)
    {
        logger.write(ELogLevel::TRACE, str, loc);
    }

    void LoggingContext::trace(std::string_view const&              str,
                               std::initializer_list<StrBuf> const& list,
                               std::source_location const&          loc)
    {
        logger.write(ELogLevel::TRACE, str, list, loc);
    }

    void LoggingContext::write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
    {
        logger.write(level, str, loc);
    }

    void LoggingContext::write(ELogLevel                            level,
                               std::string_view const&              str,
                               std::initializer_list<StrBuf> const& list,
                               std::source_location const&          loc)
    {
        logger.write(level, str, list, loc);
    }

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

    size_t LoggingContext::maxLogArgBytes() const { return logger.maxLogArgBytes(); }

    uint64_t LoggingContext::millisFromStart() const
    {
        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::high_resolution_clock::now().time_since_epoch())
                          .count();
        return static_cast<uint64_t>(now - start);
    }

} // namespace dmt