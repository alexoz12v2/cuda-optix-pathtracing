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

        inline constexpr bool isContinuationByte(char8_t c)
        {
            // byte of type 10--'----
            constexpr char8_t first = 0x80;
            constexpr char8_t last  = 0xBF;
            return c >= first && c <= last;
        }
        inline constexpr bool isStartOf2(char8_t c)
        {
            // code point inside [U+0080, U+07FF]
            // 2 byte char = 110xxxyy | 10yyzzzz
            constexpr char8_t mask  = 0xE0;
            constexpr char8_t value = 0xC0;
            return (c & mask) == value;
        }
        inline constexpr bool isStartOf3(char8_t c)
        {
            // code point inside [U+0800, U+FFFF]
            // 3 byte char = 1110wwww | 10xxxxyy | 10yyzzzz
            constexpr char8_t mask  = 0xF0;
            constexpr char8_t value = 0xE0;
            return (c & mask) == value;
        }
        inline constexpr bool isStartOf4(char8_t c)
        {
            // code point inside [U+010000, U+10FFFF]. U+10FFFF is the <a href="https://unicodebook.readthedocs.io/unicode.html">Last character</a>
            // 4 byte char = 11110uvv | 10vvwwww | 10xxxxyy || 10yyzzzz
            constexpr char8_t mask  = 0xF8;
            constexpr char8_t value = 0xF0;
            return (c & mask) == value;
        }
        inline constexpr bool isInvalidByte(char8_t c) { return c == 0xC0 || c == 0xC1 || (c >= 0xF5 || c <= 0xFF); }
        inline constexpr bool isValidUTF8(char8_t const ch[4])
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
        inline constexpr bool equal(char8_t const a[4], char8_t const b[4])
        {
            return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
        }

        /** @warning doesn't perform bouds checking */
        inline constexpr uint32_t computeUTF8Length(char8_t const* str, uint32_t numBytes)
        {
            uint32_t length = 0; // Number of UTF-8 characters
            uint32_t i      = 0; // Current byte position

            while (i < numBytes)
            {
                char8_t currentByte = str[i];

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
        concept PredicateC = std::is_invocable_r_v<Direction, T, char8_t[4]> && std::is_default_constructible_v<T> &&
                             requires(T t) { t.escaped(); };
    } // namespace utf8

    struct DMT_PLATFORM_API DefaultPredicate
    {
        constexpr utf8::Direction operator()(char8_t ch[4])
        {
            static_assert(u8'{' == 0x7B && u8'}' == 0x7D);
            bool const result = [this](char8_t c) {
                if (inside)
                    return c == u8'}';
                else
                    return c == u8'{';
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
        char8_t const* data;
        uint32_t       len;
        uint32_t       numBytes;
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
        m_start(reinterpret_cast<char8_t const*>(&_arr[0])),
        m_numBytes(N - 1)
        {
            advance();
        }

        inline constexpr FormatString(std::string_view str) :
        m_start(reinterpret_cast<char8_t const*>(str.data())),
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
                        if (pair.ch[0] != u8'}')
                        {
                            m_lastBytes -= pair.numBytes;
                            m_len--;
                            m_numBytes += pair.numBytes;
                            argFinished = true;
                        }
                        else
                            potentialExit = false;
                    }
                    else if (!escaped && pair.ch[0] == u8'{')
                        escaped = true;
                    else if (!escaped && pair.ch[0] == u8'}')
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
                    if (pair.ch[0] == u8'{')
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
            char8_t  ch[4];
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

        char8_t const* m_start;
        uint32_t       m_len       = 0;
        uint32_t       m_lastBytes = 0;
        uint32_t       m_numBytes;
        Pred           stop;
        bool           m_insideArg = false;
        bool           m_dirty     = false;
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
        char8_t const*       data;
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
        requires(std::is_invocable_v<UTF8Formatter<Ts>, Ts const&, char8_t*, uint32_t&, uint32_t&> && ...)
    inline constexpr LogRecord createRecord(
        FormatString<>              fmt,
        ELogLevel                   _level,
        char8_t*                    _buffer,
        uint32_t                    _bufferSize,
        char8_t*                    _argBuffer,
        uint32_t                    _argBufferSize,
        std::tuple<Ts...> const&    _params,
        LogLocation const&          _phyLoc,
        std::source_location const& _loc)
    {
        LogRecord record{_buffer, _phyLoc, _loc, _level, 0, 0};
        char8_t*  argBufPtr     = _argBuffer;
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
            _buffer[0] = u8'\n';
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

    /**
     * Obtain from a `ELogLevel` its string representation (in read only memory) as a `std::string_view`
     * @param level log level
     * @return stringified log level
     */
    inline constexpr std::string_view stringFromLevel(ELogLevel level)
    {
        using namespace std::string_view_literals;
        constexpr std::array<std::string_view, toUnderlying(ELogLevel::NONE)>
            strs{"TRACE"sv, "LOG  "sv, "WARN "sv, "ERROR"sv};
        return strs[toUnderlying(level)];
    }

    /**
     * Namespace containing ASCII color codes for console colored output
     */
    namespace logcolor {
        /**
         * ASCII sequence to reset the color of the terminal. called at the end of `write` of `ConsoleLogger`
         */
        inline constexpr std::string_view const reset = "\033[0m";

        /**
         *  ASCII sequence for the red color
         */
        inline constexpr std::string_view const red = "\033[31m";

        /**
         *  ASCII sequence for the green color
         */
        inline constexpr std::string_view const green = "\033[32m";

        /**
         *  ASCII sequence for a bright tint of yellow
         */
        inline constexpr std::string_view const brightYellow = "\033[93m";

        /**
         *  ASCII sequence for a darker tint of yellow
         */
        inline constexpr std::string_view const darkYellow = "\033[33m";

        /**
         *  ASCII sequence for a greyish green
         */
        inline constexpr std::string_view const greyGreen = "\033[38;5;102m";

        /**
         *  ASCII sequence for the blue color
         */
        inline constexpr std::string_view const blue = "\033[34m";

        /**
         *  ASCII sequence for the magenta color
         */
        inline constexpr std::string_view const magenta = "\033[35m";

        /**
         *  ASCII sequence for the cyan color
         */
        inline constexpr std::string_view const cyan = "\033[36m";

        /**
         *  ASCII sequence for a bold-like white color
         */
        inline constexpr std::string_view const brightWhite = "\033[97m";

        /**
         * Extract the terminal.color from an `ELogLevel`
         * @param level log level
         * @return ASCII color sequence for console logging
         */
        inline constexpr std::string_view colorFromLevel(ELogLevel level)
        {
            assert(level != ELogLevel::NONE);
            constexpr std::array<std::string_view, toUnderlying(ELogLevel::NONE)> colors{greyGreen, brightWhite, darkYellow, red};
            return colors[toUnderlying(level)];
        }
    } // namespace logcolor

    /**
     * Enum signaling which type of logger are we using
     */
    enum class ELogDisplay : uint8_t
    {
        Console,
        WindowPanel,
        Forward,
        Count,
    };

    template <std::integral I>
    inline constexpr char const* defaultFormatter()
    {
        if constexpr (std::is_signed_v<I> && sizeof(I) <= 4)
            return "%d";
        if constexpr (std::is_signed_v<I>)
            return "%lld";
        if constexpr (std::is_unsigned_v<I> && sizeof(I) <= 4)
            return "%u";
        else
            return "%zu";
    }

    /**
     * Class whose purpose is to convert to a string representation whichever types are to be supported by default in the
     * `CircularOStringStream` formatting facilities`. Uses ASCII
     */
    struct StrBuf
    {
        /**
         * Basic constructor which initializes memberwise
         * @param str pointer to cstring
         * @param len length of the pointed string, excluding the '\0' (expected positive)
         *
         */
        inline constexpr StrBuf(char const* str, int32_t len) : str(str), len(len) {}

        /**
         * Constructor which initializes from a \0 terminated string with strlen
         * @param str
         */
        inline StrBuf(char const* str) : str(str), len(static_cast<int32_t>(std::strlen(str))) {}

        /**
         * Converting constructor from a string_view. NOT `explicit` on purpose
         * @param view
         */
        inline constexpr StrBuf(std::string_view const& view) :
        str(view.data()),
        len(static_cast<int32_t>(view.length()))
        {
        }

        /**
         * Converting constructor from a string_view. NOT `explicit` on purpose
         * @param view
         */
        inline constexpr StrBuf(std::string const& view) : str(view.c_str()), len(static_cast<int32_t>(view.length()))
        {
        }

        /**
         * Converting constructor for formatting booleans
         * @param b
         */
        inline constexpr StrBuf(bool b) : StrBuf(b ? strue : sfalse) {}

        /**
         * Constructor from a floating point value. If the format string is reasonable, it shouldn't allocate and use
         * Small Buffer Optimization strings. It is NOT marked as `explicit` willingly. This overload should be used
         * only when the default formatting option is not desired
         * @tparam F floating point type
         * @param f  floating point value
         * @param fstr formatting string
         */
        template <std::floating_point F>
        inline constexpr StrBuf(F f, char const* fstr = "%.3g")
        {
            initialize(f, fstr);
        }

        /**
         * Constructor from an address in memory
         * @tparam P pointer type
         * @param f pointer value
         */
        template <typename P>
            requires std::is_pointer_v<P>
        inline constexpr StrBuf(P f)
        {
            initialize(reinterpret_cast<uintptr_t>(f), "0x%" PRIXPTR);
        }

        /**
         * Constructor from an integral value. If the format string is reasonable, it shouldn't allocate
         * @tparam I
         * @param i
         * @param fstr
         */
        template <std::integral I>
        inline constexpr StrBuf(I i, char const* fstr = defaultFormatter<I>())
        {
            initialize(i, fstr);
        }

        union
        {
            /**
             * pointer to string content of the argument, when it is not a number
             */
            char const* str;

            /**
             * small buffer containing stringified representation of a number
             */
            char buf[16];
        };

        /**
         * length of the string representation of the argument. If < 0, it means that the argument is a number stored in
         * `buf`, otherwise, `str` is the active member of the union { str, buf }. Its absolute value always holds the
         * length of the representation
         */
        int32_t len;

    private:
        /**
         * stringified representation of "true"
         */
        static constexpr std::string_view strue = "true";

        /**
         * stringified representation of "false"
         */
        static constexpr std::string_view sfalse = "false";

        /**
         * function which uses the `snprintf` method to format a number into a Small Buffer string
         * @tparam T integral or floating point type
         * @param value value to be converted into a string format
         * @param fstr formatting string
         * @return nothing, as it initializes `buf` and `len` member variables
         */
        template <typename T>
            requires(std::integral<T> || std::floating_point<T>)
        constexpr void initialize(T value, char const* fstr)
        {
            // SBO optimized string
            std::string s(15, '\0');
            auto        written = std::snprintf(s.data(), s.size(), fstr, value);
            assert(written > 0 && "invalid formatting string");
            assert(s.capacity() == 15 && "String was allocated");
            s.resize(static_cast<uint32_t>(written));
            auto sz = static_cast<int32_t>(s.size());
            std::memcpy(buf, s.data(), static_cast<uint32_t>(sz));
            buf[sz] = '\0';
            len     = -sz;
        }
    };

    class DMT_PLATFORM_API BaseAsyncIOManager
    {
    public:
        static inline constexpr uint32_t numAios = 4;
        static_assert(numAios < 5);
        static inline constexpr uint32_t lineSize    = 2048;
        static inline constexpr uint32_t maxAttempts = 10;

    protected:
        struct Line
        {
            char buf[lineSize];
        };
    };

#if defined(DMT_OS_LINUX)

    struct DMT_PLATFORM_API alignas(8) AioSpace
    {
        unsigned char bytes[256];
    };

    class DMT_PLATFORM_API LinuxAsyncIOManager : public BaseAsyncIOManager
    {
    public:
        LinuxAsyncIOManager();
        LinuxAsyncIOManager(LinuxAsyncIOManager const&) = delete;
        LinuxAsyncIOManager(LinuxAsyncIOManager&& other) noexcept;

        LinuxAsyncIOManager& operator=(LinuxAsyncIOManager const&) = delete;
        LinuxAsyncIOManager& operator=(LinuxAsyncIOManager&& other) noexcept;
        ~LinuxAsyncIOManager() noexcept;

        // Enqueue IO work to either STDOUT or STDERR
        // teh work should NOT have the
        bool     enqueue(uint32_t idx, size_t size);
        uint32_t findFirstFreeBlocking();
        char*    operator[](uint32_t idx);

        // Poll for completion of IO operations
        void sync() const;

    private:
        // Helper method to initialize the AIO control blocks
        void initAio();
        void cleanup() noexcept;

        AioSpace*     m_aioQueue;
        Line*         m_lines;
        unsigned char padding[44];
    };

#elif defined(DMT_OS_WINDOWS)

    struct DMT_PLATFORM_API alignas(8) AioSpace
    {
        unsigned char bytes[32];
    };

    class DMT_PLATFORM_API WindowsAsyncIOManager : public BaseAsyncIOManager
    {
    public:
        WindowsAsyncIOManager();
        WindowsAsyncIOManager(WindowsAsyncIOManager const&)            = delete;
        WindowsAsyncIOManager& operator=(WindowsAsyncIOManager const&) = delete;
        WindowsAsyncIOManager(WindowsAsyncIOManager&&) noexcept;
        WindowsAsyncIOManager& operator=(WindowsAsyncIOManager&&) noexcept;
        ~WindowsAsyncIOManager() noexcept;

        uint32_t findFirstFreeBlocking();
        bool     enqueue(int32_t idx, size_t size);
        char*    operator[](uint32_t idx);

    private:
        void    sync();
        void    initAio();
        int32_t waitForEvents(uint32_t timeout, bool waitAll);
        void    cleanup() noexcept;

        void*         m_hStdOut = nullptr;
        void*         m_hBuffer[numAios]{};
        AioSpace*     m_aioQueue;
        Line*         m_lines;
        unsigned char m_padding[8];
    };
#endif
    /**
     * Class which formats all the given arguments into a local buffer
     */
    class DMT_PLATFORM_API CircularOStringStream
    {
    public:
        CircularOStringStream();
        CircularOStringStream(CircularOStringStream const&) = delete;
        CircularOStringStream(CircularOStringStream&&) noexcept;
        CircularOStringStream& operator=(CircularOStringStream const&) = delete;
        CircularOStringStream& operator=(CircularOStringStream&&) noexcept;
        ~CircularOStringStream() noexcept;

        /**
         * fixed size of the circular buffer. Could be made a template param
         */
        static constexpr uint32_t bufferSize = 4096;

        /**
         * Inserts the string into the buffer
         * @warning if the buffer gets full, in debug it should crash because of the `assert(false)`, while
         * in release it silently wraps around and begins overwriting the buffer. Hence, be sure to call `clear`
         * once in a while
         * @param buf `StrBuf` containing the string to insert
         * @return itself for operator concatenation
         */
        CircularOStringStream& operator<<(StrBuf const& buf);

        /**
         * Inserts the character into the buffer
         * @warning if the buffer gets full, in debug it should crash because of the `assert(false)`, while
         * in release it silently wraps around and begins overwriting the buffer. Hence, be sure to call `clear`
         * once in a while
         * @param c character to insert into the buffer
         * @return itself for operator concatenation
         */
        CircularOStringStream& operator<<(char const c);

        /**
         * Function to set `m_pos` to 0 and `m_buffer[0] = '\0'`
         */
        void clear();

        /**
         * From the format string and the list of arguments, accumulate a formatted string into the buffer
         * @param formatStr format string
         * @param args arguments for the format string
         */
        void logInitList(char const* formatStr, std::initializer_list<StrBuf> const& args);

        /**
         * getter which '\0' terminates the string at `m_pos` and returns a `std::string_view` from it
         * @return string view from the buffer, from 0 (inclusive) to m_pos ('\0')
         */
        std::string_view str();

        size_t maxLogArgBytes() const { return bufferSize; }

    private:
        /**
         * character buffer
         */
        char* m_buffer; // zero initialisation

        /**
         * Indicator of the first free character
         */
        uint32_t m_pos;
    };

    // clang-format off
/**
 * Concept which lists all the functions and members a logger needs to implement
 * @tparam T type to check
 */
template <typename T>
concept LogDisplay = requires(T t)
{
    typename T::Traits;
    requires std::is_same_v<std::remove_cvref_t<decltype(T::Traits::displayType)>, ELogDisplay>;
    requires requires (ELogLevel level, std::string_view str, const std::source_location loc)
    {
        {t.setLevel(level)} -> std::same_as<void>;
        {t.enabled(level)} -> std::same_as<bool>;
        {t.logEnabled()} -> std::same_as<bool>;
        {t.errorEnabled()} -> std::same_as<bool>;
        {t.traceEnabled()} -> std::same_as<bool>;
        {t.warnEnabled()} -> std::same_as<bool>;
        {t.write(level, str, loc)} -> std::same_as<void>;
        {t.log(str, loc)} -> std::same_as<void>;
        {t.error(str, loc)} -> std::same_as<void>;
        {t.warn(str, loc)} -> std::same_as<void>;
        {t.trace(str, loc)} -> std::same_as<void>;
        requires requires(std::initializer_list<StrBuf> const &list)
        {
            {t.write(level, str, list, loc)} -> std::same_as<void>;
            {t.log(str, list, loc)} -> std::same_as<void>;
            {t.error(str, list, loc)} -> std::same_as<void>;
            {t.warn(str, list, loc)} -> std::same_as<void>;
            {t.trace(str, list, loc)} -> std::same_as<void>;
        };
    };
};
    // clang-format on

    // TODO ForceInline
    template <typename Derived>
    class InterfaceLogger
    {
    public:
        /**
         * Function which performs logging with the `LOG` log level, only if `m_level` is at least `LOG`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void log(std::string_view const&     str,
                                 std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::LOG, str, loc);
        }

        /**
         * Function which performs logging with the `ERROR` log level, only if `m_level` is at least `LOG`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void error(std::string_view const&     str,
                                   std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::ERR, str, loc);
        }

        /**
         * Function which performs logging with the `ERROR` log level, only if `m_level` is at least `WARN`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void warn(std::string_view const&     str,
                                  std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::WARNING, str, loc);
        }

        /**
         * Function which performs logging with the `ERROR` log level, only if `m_level` is at least `TRACE`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void trace(std::string_view const&     str,
                                   std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::TRACE, str, loc);
        }

        /**
         * Function which formats the given arguments into a format string, to then print it through the logger, only if
         * the logging level is at least `LOG`
         * @param str format string
         * @param list list of arguments
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void log(std::string_view const&              str,
                                 std::initializer_list<StrBuf> const& list,
                                 std::source_location const&          loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::LOG, str, list, loc);
        }

        /**
         * Function which formats the given arguments into a format string, to then print it through the logger, only if
         * the logging level is at least `ERROR`
         * @param str format string
         * @param list list of arguments
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void error(std::string_view const&              str,
                                   std::initializer_list<StrBuf> const& list,
                                   std::source_location const&          loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::ERR, str, list, loc);
        }

        /**
         * Function which formats the given arguments into a format string, to then print it through the logger, only if
         * the logging level is at least `WARN`
         * @param str format string
         * @param list list of arguments
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void warn(std::string_view const&              str,
                                  std::initializer_list<StrBuf> const& list,
                                  std::source_location const&          loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::WARNING, str, list, loc);
        }

        /**
         * Function which formats the given arguments into a format string, to then print it through the logger, only if
         * the logging level is at least `TRACE`
         * @param str format string
         * @param list list of arguments
         * @param loc source location of the caller, auto calculated
         */
        DMT_FORCEINLINE void trace(std::string_view const&              str,
                                   std::initializer_list<StrBuf> const& list,
                                   std::source_location const&          loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::TRACE, str, list, loc);
        }
    };

    /**
     * CRTP base class for a logger, implementing redundant functions like `log`, `error`, ... using the `Derived`
     * `write` function
     * @tparam Derived type of the derived logger class
     */
    template <typename Derived>
    class BaseLogger : public InterfaceLogger<Derived>
    {
    public:
        /**
         * explicit constructor for the base logger starting from the desired level
         * @param level desired log level
         */
        explicit BaseLogger(ELogLevel level = ELogLevel::LOG) : m_level(level) {};

        /**
         * Setter for the `m_level`
         * @param level new level
         */
        void setLevel(ELogLevel level) { m_level = level; }

        /**
         * check if the given log level is enabled
         * @param level log level
         * @return boolean indicating whether the given log level is enabled
         */
        [[nodiscard]] bool enabled(ELogLevel level) const { return m_level >= level; }

        /**
         * Checks if the `LOG` log level is enabled
         * @return boolean indicating whether the `LOG` log level is enabled
         */
        [[nodiscard]] bool logEnabled() { return static_cast<Derived*>(this)->enabled(ELogLevel::LOG); }

        /**
         * Checks if the `ERROR` log level is enabled
         * @return boolean indicating whether the `ERROR` log level is enabled
         */
        [[nodiscard]] bool errorEnabled() { return static_cast<Derived*>(this)->enabled(ELogLevel::ERR); }

        /**
         * Checks if the `TRACE` log level is enabled
         * @return boolean indicating whether the `TRACE` log level is enabled
         */
        [[nodiscard]] bool traceEnabled() { return static_cast<Derived*>(this)->enabled(ELogLevel::TRACE); }

        /**
         * Checks if the `WARN` log level is enabled
         * @return boolean indicating whether the `WARN` log level is enabled
         */
        [[nodiscard]] bool warnEnabled() { return static_cast<Derived*>(this)->enabled(ELogLevel::WARNING); }
        /**
         * Checks if the `LOG` log level is enabled
         * @return boolean indicating whether the `LOG` log level is enabled
         */
        [[nodiscard]] bool logEnabled() const { return static_cast<Derived const*>(this)->enabled(ELogLevel::LOG); }

        /**
         * Checks if the `ERROR` log level is enabled
         * @return boolean indicating whether the `ERROR` log level is enabled
         */
        [[nodiscard]] bool errorEnabled() const { return static_cast<Derived const*>(this)->enabled(ELogLevel::ERR); }

        /**
         * Checks if the `TRACE` log level is enabled
         * @return boolean indicating whether the `TRACE` log level is enabled
         */
        [[nodiscard]] bool traceEnabled() const { return static_cast<Derived const*>(this)->enabled(ELogLevel::TRACE); }

        /**
         * Checks if the `WARN` log level is enabled
         * @return boolean indicating whether the `WARN` log level is enabled
         */
        [[nodiscard]] bool warnEnabled() const
        {
            return static_cast<Derived const*>(this)->enabled(ELogLevel::WARNING);
        }

    protected:
        /**
         * Log Level. It is checked whether to format and print to the logger display or not
         */
        ELogLevel m_level;
    };

    /**
     * Size of the type erased encapsulated class `m_asyncIOClass`
     */
    inline constexpr uint32_t asyncIOClassSize = 64;

    // clang-format off
template <typename T>
concept AsyncIOManager = requires(T t) {
    requires std::is_constructible_v<T>;
    requires std::derived_from<T, BaseAsyncIOManager>;
    requires std::is_standard_layout_v<T>;
    requires std::movable<T>;
    requires sizeof(T) == asyncIOClassSize;
    requires alignof(T) == 8;
    { t[3u] } -> std::convertible_to<char *>;
    { t.findFirstFreeBlocking() } -> std::convertible_to<uint32_t>;
    { t.enqueue(3u, 3u) } -> std::same_as<bool>;
};
    // clang-format on

    /**
     * Implementation of the `AsyncIOManager` concept which does nothing. Used to fill moved-from
     * `ConsoleLogger` objects
     */
    class alignas(8) DMT_PLATFORM_API NullAsyncIOManager : public BaseAsyncIOManager
    {
    public:
        char* operator[](uint32_t i) { return m_padding; }

        uint32_t findFirstFreeBlocking() const { return 0; }

        bool enqueue(uint32_t idx, size_t sz) const { return true; }

    private:
        char m_padding[asyncIOClassSize]{};
    };


#if defined(DMT_OS_LINUX)
    static_assert(AsyncIOManager<LinuxAsyncIOManager>);
#elif defined(DMT_OS_WINDOWS)
    static_assert(AsyncIOManager<WindowsAsyncIOManager>);
#endif
    static_assert(AsyncIOManager<NullAsyncIOManager>);

    class DMT_PLATFORM_API ConsoleLogger;
    extern template BaseLogger<ConsoleLogger>;

    /**
     * Class implementing basic console logging while making use of the async IO facilities of the Windows and Linux
     * Operating system.
     * Note: OS API usage is not beneficial to performance here, but it's for learning and reference
     */
    class DMT_PLATFORM_API ConsoleLogger
    {
    public:
        ELogLevel level;

    public:
        // -- Types --
        /**
         * defining some properties as mandated by the `LogDisplay` concept
         */
        struct Traits
        {
            static constexpr ELogDisplay displayType = ELogDisplay::Console;
        };


        // -- Constructors/Copy Control --
        /**
         * Factory Method to create a logger. This allows us to have a default template parameter on
         * construction while still being able to modify it in scenarios like testing
         * @tparam T `AsyncIOManager` implementation class
         * @param level minimum log level
         * @return `ConsoleLogger` instance
         */
        template <AsyncIOManager T =
#if defined(DMT_OS_LINUX)
                      LinuxAsyncIOManager
#elif defined(DMT_OS_WINDOWS)
                      WindowsAsyncIOManager
#else
#error "what"
#endif
                  >
        static ConsoleLogger create(ELogLevel level = ELogLevel::LOG)
        {
            ConsoleLogger logger{level};
            assert(reinterpret_cast<std::uintptr_t>(logger.m_asyncIOClass) % alignof(T) == 0);
            std::construct_at<T>(reinterpret_cast<T*>(logger.m_asyncIOClass));
            logger.m_IOClassInterface.tryAsyncLog =
                [](unsigned char*          pClazz,
                   ELogLevel               levell,
                   std::string_view const& date,
                   std::string_view const& fileName,
                   std::string_view const& functionName,
                   uint32_t                line,
                   std::string_view const& levelStr,
                   std::string_view const& content) {
                T&       clazz   = *reinterpret_cast<T*>(pClazz);
                uint32_t freeIdx = clazz.findFirstFreeBlocking();
                int32_t  sz      = std::snprintf(clazz[freeIdx],
                                           T::lineSize,
                                           "%s[%s %s:%s:%u] %s <> %s%s\n\0",
                                           logcolor::colorFromLevel(levell).data(),
                                           date.data(),
                                           fileName.data(),
                                           functionName.data(),
                                           line,
                                           levelStr.data(),
                                           content.data(),
                                           logcolor::reset.data());
                assert(sz > 0 && "could not log to buffer");
                return clazz.enqueue(freeIdx, static_cast<uint32_t>(sz));
            };
            logger.m_IOClassInterface.destructor = [](unsigned char* pClazz) {
                std::destroy_at<T>(reinterpret_cast<T*>(pClazz));
            };
            return logger;
        }

        ConsoleLogger(ConsoleLogger const&)            = delete;
        ConsoleLogger& operator=(ConsoleLogger const&) = delete;

        /**
         * Move constructor which locks the mutes of the passed parameter and acquires the interface
         * and bytes of the implementation class
         * @param other
         */
        ConsoleLogger(ConsoleLogger&& other);

        /**
         * Move assignment which frees the current implementation class,
         * locks the mutes of the passed parameter and acquires the interface
         * and bytes of the implementation class
         * @param other
         */
        ConsoleLogger& operator=(ConsoleLogger&& other);

        /**
         * Destructor which is manually calling the encapsulated class' destructor
         */
        ~ConsoleLogger();

        // -- Functions  --
        /**
         * function to write to LogDisplay, only if there's the appropriate log level
         * @param level log level desired for the string
         * @param str input string
         * @param loc source location to use to create a prefix
         */
        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc);

        /**
         * function to write to LogDisplay, only if there's the appropriate log level, with arguments and format string
         * @param level log level desired for the formatted string
         * @param str format string
         * @param list arguments for the format string
         * @param loc source location used to create a prefix
         */
        void write(ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc);

        template <AsyncIOManager T>
        std::remove_cvref_t<T>& getInteralAs()
        {
            return *reinterpret_cast<std::remove_cvref_t<T>*>(&m_asyncIOClass);
        }

        size_t maxLogArgBytes() const;

    private:
        // -- Constructors --
        explicit ConsoleLogger(ELogLevel level) : level(level) {}

        // -- Types --
        /**
         * Interface type calling into the type erased class implementation
         */
        struct DMT_PLATFORM_API Table
        {
            bool (*tryAsyncLog)(unsigned char*          pClazz,
                                ELogLevel               level,
                                std::string_view const& date,
                                std::string_view const& fileName,
                                std::string_view const& functionName,
                                uint32_t                line,
                                std::string_view const& levelStr,
                                std::string_view const& content) =
                [](unsigned char*          pClazz,
                   ELogLevel               level,
                   std::string_view const& date,
                   std::string_view const& fileName,
                   std::string_view const& functionName,
                   uint32_t                line,
                   std::string_view const& levelStr,
                   std::string_view const& content) { return true; };
            void (*destructor)(unsigned char* pClazz) = [](unsigned char* pClazz) {};
        };

        // -- Constants --
        /**
         * length of the buffer used to create
         */
        static inline constexpr uint32_t timestampMax = 64;

        // -- Function Members --
        /**
         * Private function member used to implement move semantics. It takes the interface and
         * bytes of the io manager internal class
         * @param other
         */
        void stealResourcesFrom(ConsoleLogger&& other);

        /**
         * Helper function to format and print the log message
         * @param level desired log level. Used to determine the color of the console output
         * @param date formatted timestamp string to insert in the console output
         * @param fileName string filename relative to the project directory
         * @param functionName name of the function coming from the `std::source_location`
         * @param line line number coming from the `std::source_location`
         * @param levelStr stringified representation of the log level to insert at the end of the prefix
         * @param content formatted string to insert in the console output
         */
        void logMessage(ELogLevel               level,
                        std::string_view const& date,
                        std::string_view const& fileName,
                        std::string_view const& functionName,
                        uint32_t                line,
                        std::string_view const& levelStr,
                        std::string_view const& content);

        /**
         * Async IO version of the `logMessage` function, using, as an exercise, for no practical reason, the asynchronous
         * IO api given by the supported platforms. If, for any reason, the IO operation fails, it falls back to the
         * `logMessage` function
         * @param level desired log level. Used to determine the color of the console output
         * @param date formatted timestamp string to insert in the console output
         * @param fileName string filename relative to the project directory
         * @param functionName name of the function coming from the `std::source_location`
         * @param line line number coming from the `std::source_location`
         * @param levelStr stringified representation of the log level to insert at the end of the prefix
         * @param content formatted string to insert in the console output
         */
        void logMessageAsync(ELogLevel               level,
                             std::string_view const& date,
                             std::string_view const& fileName,
                             std::string_view const& functionName,
                             uint32_t                line,
                             std::string_view const& levelStr,
                             std::string_view const& content);

        /**
         * Helper function remove the `PROJECT_SOURCE_DIR` from the input string view,
         * There's an `assert` which checks whether the given path is a subdirectory of the project
         * @param fullPath full path from a `std::source_location`
         */
        static void carveRelativeFileName(std::string_view& fullPath);

        /**
         * Helper function to get the current timestamp from the `std::system_clock`, then formatted with `<ctime>`
         * @return formatted timestamp
         */
        [[nodiscard]] std::string_view getCurrentTimestamp();

        // -- Members --
        /**
         * stream like object to accumulate the formatted string when parsing the arguments and the format string
         */
        CircularOStringStream m_oss;

        /**
         * buffer used to hold an instance of the timestamp when printing
         * @warning this is to remove and dynamically allocate when multithreading this
         */
        char m_timestampBuf[timestampMax]{};

        /**
         * type erased class to access OS-specific functionalities
         * @warning this is to remove and synamically allocate when multithreading this
         */
        alignas(8) mutable unsigned char m_asyncIOClass[asyncIOClassSize]{};

        /**
         * Table of functions to handle the type erased class `m_asyncIOClass`
         */
        Table m_IOClassInterface;

        /**
         * Mutex to ensure thread-safety for write methods
         */
        static inline std::mutex s_writeMutex;
    };

    class DMT_PLATFORM_API LoggingContext : public InterfaceLogger<LoggingContext>
    {
    public:
        LoggingContext();

        /**
         * Setter for the `m_level`
         * @param level new level
         * @warning Purposefully name hiding the `BaseLogger`
         */
        void setLevel(ELogLevel level) { logger.level = level; }

        void log(std::string_view const& str, std::source_location const& loc = std::source_location::current());
        void log(std::string_view const&              str,
                 std::initializer_list<StrBuf> const& list,
                 std::source_location const&          loc = std::source_location::current());

        void warn(std::string_view const& str, std::source_location const& loc = std::source_location::current());
        void warn(std::string_view const&              str,
                  std::initializer_list<StrBuf> const& list,
                  std::source_location const&          loc = std::source_location::current());

        void error(std::string_view const& str, std::source_location const& loc = std::source_location::current());
        void error(std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc = std::source_location::current());

        void trace(std::string_view const& str, std::source_location const& loc = std::source_location::current());
        void trace(std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc = std::source_location::current());

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str string to output
         * @param loc location of the log
         */
        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc);

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str format string
         * @param list list of arguments which will be used to create the final string
         * @param loc location of the log
         */
        void write(ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc);

        /**
         * CRTP overridden function to check if the true underlying logger is enabled on the log level
         * @param level log level requested
         * @return bool signaling whether the requested log level is enabled
         */
        bool enabled(ELogLevel level) const { return level >= logger.level; }

        bool traceEnabled() const { return enabled(ELogLevel::TRACE); }

        bool logEnabled() const { return enabled(ELogLevel::LOG); }

        bool warnEnabled() const { return enabled(ELogLevel::WARNING); }

        bool errorEnabled() const { return enabled(ELogLevel::ERR); }

        void dbgTraceStackTrace();

        void dbgErrorStackTrace();

        size_t maxLogArgBytes() const;

        uint64_t millisFromStart() const;

        ConsoleLogger logger;
        int64_t       start;
    };


    template <std::floating_point T>
    struct UTF8Formatter<T>
    {
        inline constexpr void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
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
        inline constexpr void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
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
        inline constexpr void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
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

} // namespace dmt
