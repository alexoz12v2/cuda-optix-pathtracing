#pragma once

#include <platform/platform-macros.h>
#include <platform/platform-utils.h>

#include <array>
#include <chrono>
#include <concepts>
#include <format>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <cassert>
#include <cinttypes>
#include <compare>
#include <cstdint>
#include <cstring>

namespace dmt {
    /**
     * @defgroup UTF-8 byte query utils.
     * @note In the following, consider a code point of type `U+uvwxyz`, in which all letters but `u` are 4 bits,
     * while `u` is only 1 bit, as it can be either 0 or 1. * <a href="https://en.wikipedia.org/wiki/UTF-8#Description">Reference</a>
     * @{
     */
    namespace utf8 {
        enum class Direction : uint8_t
        {
            eNone = 0,
            eForward,
            eBackward
        };

        inline constexpr bool isContinuationByte(char c)
        {
            // byte of type 10--'----
            constexpr char first = 0x80;
            constexpr char last  = 0xBF;
            return c >= first && c <= last;
        }
        inline constexpr bool isStartOf2(char c)
        {
            // code point inside [U+0080, U+07FF]
            // 2 byte char = 110xxxyy | 10yyzzzz
            constexpr char mask  = 0xE0;
            constexpr char value = 0xC0;
            return (c & mask) == value;
        }
        inline constexpr bool isStartOf3(char c)
        {
            // code point inside [U+0800, U+FFFF]
            // 3 byte char = 1110wwww | 10xxxxyy | 10yyzzzz
            constexpr char mask  = 0xF0;
            constexpr char value = 0xE0;
            return (c & mask) == value;
        }
        inline constexpr bool isStartOf4(char c)
        {
            // code point inside [U+010000, U+10FFFF]. U+10FFFF is the <a href="https://unicodebook.readthedocs.io/unicode.html">Last character</a>
            // 4 byte char = 11110uvv | 10vvwwww | 10xxxxyy || 10yyzzzz
            constexpr char mask  = 0xF8;
            constexpr char value = 0xF0;
            return (c & mask) == value;
        }
        inline constexpr bool isInvalidByte(char c) { return c == 0xC0 || c == 0xC1 || (c >= 0xF5 || c <= 0xFF); }
        inline constexpr bool isValidUTF8(char const ch[4])
        {
            if (isInvalidByte(ch[0]) || isInvalidByte(ch[1]) || isInvalidByte(ch[2]) || isInvalidByte(ch[3]))
                return false;

            if (isStartOf4(ch[0]))
                return (ch[1] & 0xC0) == 0x80 && (ch[2] & 0xC0) == 0x80 &&
                       (ch[3] & 0xC0) == 0x80; // 11110uvv | 10vvwwww | 10xxxxyy | 10yyzzzz (`U+uvwxyz`)
            else if (isStartOf3(ch[0]))
                return (ch[1] & 0xC0) == 0x80 && (ch[2] & 0xC0) == 0x80; // 1110wwww | 10xxxxyy | 10yyzzzz (`U+uvwxyz`)
            else if (isStartOf2(ch[0]))
                return (ch[1] & 0xC0) == 0x80;   // 110xxxyy | 10yyzzzz (`U+uvwxyz`)
            else if (!isContinuationByte(ch[0])) // "Basic Latin" block [U+0000, U+007F]
                return (ch[0] & 0x80) == 0x00;   // 0yyyzzzz (`U+uvwxyz`)
            else
                return false;
        }
        inline constexpr bool equal(char const a[4], char const b[4])
        {
            return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
        }

        /** @warning doesn't perform bouds checking */
        inline constexpr uint32_t computeUTF8Length(char const* str, uint32_t numBytes)
        {
            uint32_t length = 0; // Number of UTF-8 characters
            uint32_t i      = 0; // Current byte position

            while (i < numBytes)
            {
                char currentByte = str[i];

                if (isInvalidByte(currentByte))
                    return 0; // Return 0 or throw an exception if you prefer error handling for invalid bytes

                // Determine the number of bytes in the current UTF-8 character
                if (isStartOf4(currentByte))
                {
                    if (i + 3 < numBytes && isContinuationByte(str[i + 1]) && isContinuationByte(str[i + 2]) &&
                        isContinuationByte(str[i + 3]))
                        i += 4;
                    else
                        return 0; // Invalid 4-byte sequence
                }
                else if (isStartOf3(currentByte))
                {
                    if (i + 2 < numBytes && isContinuationByte(str[i + 1]) && isContinuationByte(str[i + 2]))
                        i += 3;
                    else
                        return 0; // Invalid 3-byte sequence
                }
                else if (isStartOf2(currentByte))
                {
                    if (i + 1 < numBytes && isContinuationByte(str[i + 1]))
                        i += 2;
                    else
                        return 0; // Invalid 2-byte sequence
                }
                else if (!isContinuationByte(currentByte)) // Single-byte character (ASCII)
                    i += 1;
                else
                    return 0; // Invalid sequence

                ++length; // Increment character count
            }

            return length;
        }
        /** @}*/

        /// returns true whenever you need to stop advancing
        template <typename T>
        concept PredicateC = std::is_invocable_r_v<Direction, T, char[4]> && std::is_default_constructible_v<T> &&
                             requires(T t) { t.escaped(); };
    } // namespace utf8

    struct DMT_PLATFORM_API DefaultPredicate
    {
        constexpr utf8::Direction operator()(char ch[4])
        {
            static_assert('{' == 0x7B && '}' == 0x7D);
            bool const result = [this](char c) {
                if (inside)
                    return c == '}';
                else
                    return c == '{';
            }(ch[0]);
            if (result)
                inside = !inside;
            return result ? (inside ? utf8::Direction::eBackward : utf8::Direction::eForward) : utf8::Direction::eNone;
        }
        constexpr void escaped() { inside = !inside; }

        bool inside = false;
    };

    /** Definition of a string view to a UTF-8 encoded string */
    struct CharRangeU8
    {
        char const* data;
        uint32_t    len;
        uint32_t    numBytes;
    };
    static_assert(std::is_trivial_v<CharRangeU8> && std::is_standard_layout_v<CharRangeU8>);

    template <utf8::PredicateC Pred = DefaultPredicate>
    class FormatString
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type      = CharRangeU8;

        template <uint32_t N>
        constexpr FormatString(char const (&_arr)[N]) :
        m_start(reinterpret_cast<char const*>(&_arr[0])),
        m_numBytes(N - 1)
        {
            advance();
        }

        inline constexpr FormatString(std::string_view str) :
        m_start(reinterpret_cast<char const*>(str.data())),
        m_numBytes(static_cast<uint32_t>(str.size()))
        {
            advance();
        }

        constexpr value_type operator*() const { return {m_start, m_len, m_lastBytes}; }

        constexpr void operator++() { advance(); }
        constexpr bool finished() const { return m_numBytes == 0 && m_dirty; }
        constexpr bool isFormatSpecifier() const { return m_insideArg; }

    private:
        constexpr void advance()
        {
            if (m_numBytes == 0)
            {
                m_dirty = true;
                return;
            }

            m_start += m_lastBytes;
            m_lastBytes             = 0;
            m_len                   = 0;
            m_insideArg             = false;
            bool     foundSpecifier = false;
            bool     escaped        = false;
            bool     potentialExit  = false;
            bool     argFinished    = false;
            bool     checkEscape    = false;
            uint32_t lastNumBytes   = 0;

            while (m_numBytes > 0 && !argFinished)
            {
                Pair pair = acquireCharacter();
                m_lastBytes += pair.numBytes;
                m_len++;
                m_numBytes -= pair.numBytes;

                if (foundSpecifier)
                {
                    if (!escaped && potentialExit)
                    {
                        if (pair.ch[0] != '}')
                        {
                            m_lastBytes -= pair.numBytes;
                            m_len--;
                            m_numBytes += pair.numBytes;
                            argFinished = true;
                        }
                        else
                            potentialExit = false;
                    }
                    else if (!escaped && pair.ch[0] == '{')
                        escaped = true;
                    else if (!escaped && pair.ch[0] == '}')
                    {
                        if (!potentialExit && m_numBytes > 0)
                            potentialExit = true;
                        else if (m_numBytes == 0)
                            argFinished = true; // shouldn't be needed
                    }
                    else if (escaped)
                    {
                        foundSpecifier = false;
                        m_insideArg    = false;
                    }
                }
                else
                {
                    if (pair.ch[0] == '{')
                    {
                        if (m_len > 1)
                        {
                            if (checkEscape)
                                checkEscape = false;
                            else if (m_numBytes > 0)
                            {
                                checkEscape  = true;
                                lastNumBytes = pair.numBytes;
                            }
                            else
                            {
                                argFinished = true;
                            }
                        }
                        else
                        {
                            assert(!argFinished && !checkEscape);
                            lastNumBytes   = 0;
                            foundSpecifier = true;
                            m_insideArg    = true;
                        }
                    }
                    else if (checkEscape)
                    {

                        m_lastBytes -= pair.numBytes + lastNumBytes;
                        m_len -= 2;
                        m_numBytes += pair.numBytes + lastNumBytes;
                        argFinished = true;
                    }
                    else
                        lastNumBytes = 0;
                }
            }
        }

        struct Pair
        {
            char     ch[4];
            uint32_t numBytes;
        };

        constexpr Pair acquireCharacter() const
        {
            if (utf8::isStartOf4(m_start[m_lastBytes]))
                return {{m_start[m_lastBytes], m_start[m_lastBytes + 1], m_start[m_lastBytes + 2], m_start[m_lastBytes + 3]},
                        4};
            if (utf8::isStartOf3(m_start[m_lastBytes]))
                return {{m_start[m_lastBytes], m_start[m_lastBytes + 1], m_start[m_lastBytes + 2], 0}, 3};
            if (utf8::isStartOf2(m_start[m_lastBytes]))
                return {{m_start[m_lastBytes], m_start[m_lastBytes + 1], 0, 0}, 2};
            return {{m_start[m_lastBytes + 0], 0, 0, 0}, 1};
        }

        char const* m_start;
        uint32_t    m_len       = 0;
        uint32_t    m_lastBytes = 0;
        uint32_t    m_numBytes;
        Pred        stop;
        bool        m_insideArg = false;
        bool        m_dirty     = false;
    };

    /**
     * Log Level enum, to check whether we should print or not, and to determine the style of the output
     * @brief log levels for logger configuration
     * @warning do not change the numbers. the implementation details depends on these exact numbers
     */
    enum class ELogLevel : uint8_t
    {
        TRACE   = 0, /** <Debug log level> */
        LOG     = 1, /** <Info log level> */
        WARNING = 2, /** <Warning log level> */
        ERR     = 3, /** <Error log level> */
        NONE    = 4, /** <Log disabled> */
        eTrace  = TRACE,
        eLog    = LOG,
        eWarn   = WARNING,
        eError  = ERR,
        eCount  = NONE,
    };

    /**
     * Spaceship operator (three-way comparison) to generate log level comparison operators
     * @param lhs log level no 1
     * @param rhs log level no 2
     * @return std::strong_ordering::{less,equivalent,greater} depending on the 2 values
     */
    inline constexpr std::strong_ordering operator<=>(ELogLevel lhs, ELogLevel rhs) noexcept
    {
        return toUnderlying(lhs) <=> toUnderlying(rhs);
    }

    // ----------------------------------------------------------------------------------------------------------------
    struct DMT_PLATFORM_API LogLocation
    {
        struct DMT_PLATFORM_API Device
        {
            int32_t blockX;
            int32_t blockY;
            int32_t blockZ;
            int32_t threadX;
            int32_t threadY;
            int32_t threadZ;
            int32_t lane;
            int32_t warp;
        };
        struct DMT_PLATFORM_API Host
        {
            uint64_t pid;
            uint64_t tid;
        };
        union DMT_PLATFORM_API U
        {
            Device dev;
            Host   host;
        };
        static constexpr int32_t hostNum = -1;

        U       loc;
        int32_t where; // if dirrerent than how then this is device id
    };

    struct DMT_PLATFORM_API LogRecord
    {
        char const*          data;
        LogLocation          phyLoc;
        std::source_location srcLoc;
        ELogLevel            level;
        uint32_t             len;
        uint32_t             numBytes;
    };

    // TODO refactor elsewhere
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid
    inline LogLocation getPhysicalLocation()
    {
        LogLocation loc;
#if defined(__CUDA_ARCH__)
        int32_t     device  = -1;
        cudaError_t err     = cudaGetDevice(&device);
        loc.where           = device;
        loc.loc.dev.blockX  = blockIdx.x;
        loc.loc.dev.blockY  = blockIdx.y;
        loc.loc.dev.blockZ  = blockIdx.z;
        loc.loc.dev.threadX = threadIdx.x;
        loc.loc.dev.threadY = threadIdx.y;
        loc.loc.dev.threadZ = threadIdx.z;
        loc.loc.dev.lane    = threadIdx.x % warpSize;
        loc.loc.dev.warp    = threadIdx.x / warpSize;
#else
        loc.where        = LogLocation::hostNum;
        loc.loc.host.pid = os::processId();
        loc.loc.host.tid = os::threadId();
#endif
        return loc;
    }

    template <typename T>
    struct UTF8Formatter;

    /** Shortcut to write `std::make_tuple` */

    template <typename... Ts>
        requires(std::is_invocable_v<UTF8Formatter<Ts>, Ts const&, char*, uint32_t&, uint32_t&> && ...)
    inline constexpr LogRecord createRecord(
        FormatString<>              fmt,
        ELogLevel                   _level,
        char*                       _buffer,
        uint32_t                    _bufferSize,
        char*                       _argBuffer,
        uint32_t                    _argBufferSize,
        std::tuple<Ts...> const&    _params,
        LogLocation const&          _phyLoc,
        std::source_location const& _loc)
    {
        LogRecord record{_buffer, _phyLoc, _loc, _level, 0, 0};
        char*     argBufPtr     = _argBuffer;
        uint32_t  totalArgBytes = 0;

        std::apply([&](auto&&... args) {
            ((UTF8Formatter<std::decay_t<decltype(args)>>{}(args, argBufPtr, totalArgBytes, _argBufferSize)), ...);
        }, _params);

        while (!fmt.finished())
        {
            if (fmt.isFormatSpecifier() && argBufPtr < _argBuffer + (totalArgBytes - _argBufferSize))
            {
                uint32_t numBytes = *reinterpret_cast<uint32_t*>(argBufPtr);
                uint32_t len      = *reinterpret_cast<uint32_t*>(argBufPtr + sizeof(uint32_t));
                memcpy(_buffer, argBufPtr + 2 * sizeof(uint32_t), numBytes);
                _buffer += numBytes;
                _bufferSize -= numBytes;
                record.len += len;
                record.numBytes += numBytes;
                argBufPtr += numBytes + 2 * sizeof(uint32_t);
            }
            else
            {
                CharRangeU8 portion     = *fmt;
                uint32_t    bytesToCopy = std::min(_bufferSize, portion.numBytes);
                memcpy(_buffer, portion.data, bytesToCopy);
                _buffer += bytesToCopy;
                _bufferSize -= bytesToCopy;
                record.len += portion.len;
                record.numBytes += bytesToCopy;
            }
            ++fmt;
        }

        if (record.numBytes < _bufferSize)
        {
            _buffer[0] = '\n';
            record.numBytes++;
            record.len++;
        }
        return record;
    }

    using LogHandlerAllocate   = void* (*)(size_t _numBytes, size_t _align);
    using LogHandlerDeallocate = void (*)(void* _ptr, size_t _numBytes, size_t _align);
    struct DMT_PLATFORM_API LogHandler
    {
        void* data;
        void (*hostFlush)(void* _data);
        bool (*hostFilter)(void* _data, LogRecord const& record);
        void (*hostCallback)(void* _data, LogRecord const& record);
        void (*hostCleanup)(LogHandlerDeallocate _dealloc, void* _data);

        // custom memory interface (internal usage only)
        LogHandlerAllocate   hostAllocate;
        LogHandlerDeallocate hostDeallocate;

        ELogLevel minimumLevel;
    };

    DMT_PLATFORM_API bool createConsoleHandler(LogHandler&          _out,
                                               LogHandlerAllocate   _alloc   = ::dmt::os::allocate,
                                               LogHandlerDeallocate _dealloc = ::dmt::os::deallocate);

    // ----------------------------------------------------------------------------------------------------------------

    template <std::floating_point T>
    struct UTF8Formatter<T>
    {
        inline constexpr void operator()(T const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            char* writePos = reinterpret_cast<char*>(_buffer + _offset + 2 * sizeof(uint32_t));
            int bytesWritten = std::snprintf(writePos, _bufferSize - _offset - 2 * sizeof(uint32_t), "%.6g", value); // Adjust precision if needed
            if (bytesWritten > 0 && static_cast<uint32_t>(bytesWritten) <= (_bufferSize - _offset - 2 * sizeof(uint32_t)))
            {
                uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
                uint32_t len      = numBytes; // Each byte corresponds to one character in this case

                *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
                *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

                _offset += 2 * sizeof(uint32_t) + numBytes;
                _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
            }
            else
            {
                _bufferSize = 0; // Insufficient buffer space
            }
        }
    };

    template <std::integral T>
    struct UTF8Formatter<T>
    {
        inline constexpr void operator()(T const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            char* writePos     = reinterpret_cast<char*>(_buffer + _offset + 2 * sizeof(uint32_t));
            int   bytesWritten = std::snprintf(writePos, _bufferSize - _offset - 2 * sizeof(uint32_t), "%d", value);

            if (bytesWritten > 0 && static_cast<uint32_t>(bytesWritten) <= (_bufferSize - _offset - 2 * sizeof(uint32_t)))
            {
                uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
                uint32_t len      = numBytes; // Each byte corresponds to one character in this case

                *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
                *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

                _offset += 2 * sizeof(uint32_t) + numBytes;
                _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
            }
            else
            {
                _bufferSize = 0; // Insufficient buffer space
            }
        }
    };

    // TODO pstd::string_view
    template <std::convertible_to<std::string_view> T>
    struct UTF8Formatter<T>
    {
        inline constexpr void operator()(T const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            std::string_view strView = value;

            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            uint32_t numBytes = static_cast<uint32_t>(strView.size());
            uint32_t len = static_cast<uint32_t>(strView.size()); // Assuming valid UTF-8 input where each character is 1 byte

            if (_bufferSize < _offset + 2 * sizeof(uint32_t) + numBytes)
                return; // Insufficient space for the string

            *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
            *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

            memcpy(_buffer + _offset + 2 * sizeof(uint32_t), strView.data(), numBytes);

            _offset += 2 * sizeof(uint32_t) + numBytes;
            _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
        }
    };

    template <typename T>
    struct UTF8Formatter<T*>
    {
        inline constexpr void operator()(T* const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            char* writePos     = reinterpret_cast<char*>(_buffer + _offset + 2 * sizeof(uint32_t));
            int   bytesWritten = std::snprintf(writePos,
                                             _bufferSize - _offset - 2 * sizeof(uint32_t),
                                             "%p",
                                             static_cast<void const*>(value));

            if (bytesWritten > 0 && static_cast<uint32_t>(bytesWritten) <= (_bufferSize - _offset - 2 * sizeof(uint32_t)))
            {
                uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
                uint32_t len      = numBytes; // Each byte corresponds to one character

                *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
                *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

                _offset += 2 * sizeof(uint32_t) + numBytes;
                _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
            }
            else
            {
                _bufferSize = 0; // Insufficient buffer space
            }
        }
    };

} // namespace dmt
