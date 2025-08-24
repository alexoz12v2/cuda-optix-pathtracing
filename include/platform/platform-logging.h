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
#include <type_traits>
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

        constexpr void     operator++() { advance(); }
        constexpr bool     finished() const { return m_numBytes == 0 && m_dirty; }
        constexpr bool     isFormatSpecifier() const { return m_insideArg; }
        constexpr uint32_t totalBytes() const { return m_numBytes + m_lastBytes; }

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
                        {
                            if (m_numBytes == 0 || m_start[m_lastBytes] == '}')
                            {
                                m_lastBytes -= pair.numBytes;
                                m_len--;
                                m_numBytes += pair.numBytes;
                                argFinished = true;
                            }
                            else
                                potentialExit = false;
                        }
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

        constexpr size_t tupleSize = sizeof...(Ts);

        // If tuple is not empty, apply formatters
        if constexpr (tupleSize > 0)
        {
            std::apply([&](auto&&... args) {
                ((UTF8Formatter<std::decay_t<decltype(args)>>{}(args, argBufPtr, totalArgBytes, _argBufferSize)), ...);
            }, _params);
        }

        uint32_t usedArgs  = 0;
        bool     earlyExit = false;

        while (!fmt.finished() && !earlyExit)
        {
            if (fmt.isFormatSpecifier() && usedArgs < tupleSize)
            {
                // There's an argument to format
                if (argBufPtr < _argBuffer + _argBufferSize)
                {
                    uint32_t numBytes = *reinterpret_cast<uint32_t*>(argBufPtr);
                    uint32_t len      = *reinterpret_cast<uint32_t*>(argBufPtr + sizeof(uint32_t));

                    if (_bufferSize >= numBytes)
                    {
                        memcpy(_buffer, argBufPtr + 2 * sizeof(uint32_t), numBytes);
                        _buffer += numBytes;
                        _bufferSize -= numBytes;
                        record.len += len;
                        record.numBytes += numBytes;
                    }

                    argBufPtr += numBytes + 2 * sizeof(uint32_t);
                    ++usedArgs;
                }
            }
            else
            {
                // Not a format specifier OR ran out of args � treat as literal
                earlyExit             = tupleSize == 0 || usedArgs >= tupleSize;
                CharRangeU8 portion   = *fmt;
                char const* src       = portion.data;
                uint32_t    remaining = earlyExit ? fmt.totalBytes() : portion.numBytes;

                while (remaining > 0 && _bufferSize > 0)
                {
                    if (remaining >= 2 && src[0] == '{' && src[1] == '{')
                    {
                        *_buffer++ = '{';
                        src += 2;
                        remaining -= 2;
                    }
                    else if (remaining >= 2 && src[0] == '}' && src[1] == '}')
                    {
                        *_buffer++ = '}';
                        src += 2;
                        remaining -= 2;
                    }
                    else
                    {
                        *_buffer++ = *src++;
                        remaining--;
                    }
                    _bufferSize--;
                    record.len++;
                    record.numBytes++;
                }
            }

            if (!earlyExit)
                ++fmt;
        }

        _buffer[record.numBytes < _bufferSize ? 0 : -1] = '\n';
        record.numBytes++;
        record.len++;

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
    template <typename T>
    T* implicit_aligned_alloc(std::size_t alignment, void*& p, std::size_t& sz)
    {
        void* aligned_ptr = p;
        if (std::align(alignment, sizeof(T), aligned_ptr, sz))
        {
            T* result = std::launder(reinterpret_cast<T*>(aligned_ptr));
            // move pointer past allocated object
            p = static_cast<std::byte*>(aligned_ptr) + sizeof(T);
            sz -= sizeof(T);
            return result;
        }
        return nullptr; // not enough space
    }

    template <typename T>
    struct UTF8Formatter;

    template <std::floating_point T>
    struct UTF8Formatter<T>
    {
        inline void operator()(T const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            void*       p  = _buffer + _offset;
            std::size_t sz = _bufferSize;

            // Align buffer for two uint32_t metadata
            void* aligned_metadata = std::align(alignof(uint32_t), 2 * sizeof(uint32_t), p, sz);
            if (!aligned_metadata || sz < 2 * sizeof(uint32_t))
            {
                _bufferSize = 0;
                return;
            }

            uint32_t* numBytesPtr = std::launder(reinterpret_cast<uint32_t*>(aligned_metadata));
            uint32_t* lenPtr      = numBytesPtr + 1;

            // Align buffer for string write after metadata
            char*       stringPos = reinterpret_cast<char*>(lenPtr + 1);
            std::size_t stringSz  = sz - 2 * sizeof(uint32_t);

            void*       aligned_string = stringPos;
            std::size_t remaining      = stringSz;
            aligned_string             = std::align(alignof(std::max_align_t), stringSz, aligned_string, remaining);
            if (!aligned_string || remaining == 0)
            {
                _bufferSize = 0;
                return;
            }

            char* writePos = reinterpret_cast<char*>(aligned_string);

            // Format float/double/long double into buffer
            int bytesWritten = std::snprintf(writePos, remaining, "%.6g", value); // Adjust precision if needed

            if (bytesWritten <= 0 || static_cast<std::size_t>(bytesWritten) > remaining)
            {
                _bufferSize = 0;
                return;
            }

            uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
            uint32_t len      = numBytes;

            *numBytesPtr = numBytes;
            *lenPtr      = len;

            // Update offsets
            _offset += (writePos - (_buffer + _offset)) + bytesWritten;
            _bufferSize -= (writePos - (_buffer + _offset)) + bytesWritten;
        }
    };


    template <typename T>
        requires(std::integral<T> || (std::is_pointer_v<T> && (!std::convertible_to<T, std::string_view>)))
    struct UTF8Formatter<T>
    {
        inline void operator()(T const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            void*       p  = _buffer + _offset;
            std::size_t sz = _bufferSize;

            // Align buffer for two uint32_t metadata
            void* aligned_metadata = std::align(alignof(uint32_t), 2 * sizeof(uint32_t), p, sz);
            if (!aligned_metadata || sz < 2 * sizeof(uint32_t))
            {
                _bufferSize = 0;
                return;
            }

            uint32_t* numBytesPtr = std::launder(reinterpret_cast<uint32_t*>(aligned_metadata));
            uint32_t* lenPtr      = numBytesPtr + 1;

            // Align buffer for string write after metadata
            char*       stringPos = reinterpret_cast<char*>(lenPtr + 1);
            std::size_t stringSz  = sz - 2 * sizeof(uint32_t);

            void*       aligned_string = stringPos;
            std::size_t remaining      = stringSz;
            aligned_string             = std::align(alignof(std::max_align_t), stringSz, aligned_string, remaining);
            if (!aligned_string || remaining == 0)
            {
                _bufferSize = 0;
                return;
            }

            char* writePos = reinterpret_cast<char*>(aligned_string);

            // Format value into buffer
            int bytesWritten = 0;
            if constexpr (std::is_pointer_v<T>)
                bytesWritten = std::snprintf(writePos, remaining, "%p", value);
            else if constexpr (sizeof(T) == 8)
                bytesWritten = std::snprintf(writePos, remaining, std::is_signed_v<T> ? "%zd" : "%zu", value);
            else
                bytesWritten = std::snprintf(writePos, remaining, std::is_signed_v<T> ? "%d" : "%u", value);

            if (bytesWritten <= 0 || static_cast<std::size_t>(bytesWritten) > remaining)
            {
                _bufferSize = 0;
                return;
            }

            uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
            uint32_t len      = numBytes;

            *numBytesPtr = numBytes;
            *lenPtr      = len;

            // Update offsets
            _offset += (writePos - (_buffer + _offset)) + bytesWritten;
            _bufferSize -= (writePos - (_buffer + _offset)) + bytesWritten;
        }
    };

    template <std::convertible_to<std::string_view> T>
    struct UTF8Formatter<T>
    {
        inline void operator()(T const& value, char* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            std::string_view sv = value; // convert to string_view

            void*       p  = _buffer + _offset;
            std::size_t sz = _bufferSize;

            // Align buffer for two uint32_t metadata
            void* aligned_metadata = std::align(alignof(uint32_t), 2 * sizeof(uint32_t), p, sz);
            if (!aligned_metadata || sz < 2 * sizeof(uint32_t))
            {
                _bufferSize = 0;
                return;
            }

            uint32_t* numBytesPtr = std::launder(reinterpret_cast<uint32_t*>(aligned_metadata));
            uint32_t* lenPtr      = numBytesPtr + 1;

            // Align buffer for string content
            char*       stringPos = reinterpret_cast<char*>(lenPtr + 1);
            std::size_t stringSz  = sz - 2 * sizeof(uint32_t);

            void*       aligned_string = stringPos;
            std::size_t remaining      = stringSz;
            aligned_string             = std::align(alignof(std::max_align_t), sv.size(), aligned_string, remaining);
            if (!aligned_string || remaining < sv.size())
            {
                _bufferSize = 0;
                return;
            }

            char* writePos = reinterpret_cast<char*>(aligned_string);

            // Copy string content
            std::memcpy(writePos, sv.data(), sv.size());

            uint32_t numBytes = static_cast<uint32_t>(sv.size());
            uint32_t len      = numBytes;

            *numBytesPtr = numBytes;
            *lenPtr      = len;

            // Update offsets
            _offset += (writePos - (_buffer + _offset)) + numBytes;
            _bufferSize -= (writePos - (_buffer + _offset)) + numBytes;
        }
    };
} // namespace dmt
