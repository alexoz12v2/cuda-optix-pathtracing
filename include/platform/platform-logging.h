#pragma once

#include "dmtmacros.h"
#include <platform/platform-macros.h>

#include <platform/platform-utils.h>

#if !defined(DMT_NEEDS_MODULE)
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
#endif

DMT_MODULE_EXPORT namespace dmt {
    /**
     * Log Level enum, to check whether we should print or not, and to determine the style of the output
     * @brief log levels for logger configuration
     */
    enum class ELogLevel : uint8_t
    {
        TRACE   = 0, /** <Debug log level> */
        LOG     = 1, /** <Info log level> */
        WARNING = 2, /** <Warning log level> */
        ERR     = 3, /** <Error log level> */
        NONE    = 4, /** <Log disabled> */
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
    struct DMT_PLATFORM_API StrBuf
    {
        /**
         * Basic constructor which initializes memberwise
         * @param str pointer to cstring
         * @param len length of the pointed string, excluding the '\0' (expected positive)
         *
         */
        constexpr StrBuf(char const* str, int32_t len) : str(str), len(len) {}

        /**
         * Constructor which initializes from a \0 terminated string with strlen
         * @param str
         */
        StrBuf(char const* str) : str(str), len(static_cast<int32_t>(std::strlen(str))) {}

        /**
         * Converting constructor from a string_view. NOT `explicit` on purpose
         * @param view
         */
        constexpr StrBuf(std::string_view const& view) : str(view.data()), len(static_cast<int32_t>(view.length())) {}

        /**
         * Converting constructor from a string_view. NOT `explicit` on purpose
         * @param view
         */
        constexpr StrBuf(std::string const& view) : str(view.c_str()), len(static_cast<int32_t>(view.length())) {}

        /**
         * Converting constructor for formatting booleans
         * @param b
         */
        constexpr StrBuf(bool b) : StrBuf(b ? strue : sfalse) {}

        /**
         * Constructor from a floating point value. If the format string is reasonable, it shouldn't allocate and use
         * Small Buffer Optimization strings. It is NOT marked as `explicit` willingly. This overload should be used
         * only when the default formatting option is not desired
         * @tparam F floating point type
         * @param f  floating point value
         * @param fstr formatting string
         */
        template <std::floating_point F>
        constexpr StrBuf(F f, char const* fstr = "%.3f")
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
        constexpr StrBuf(P f)
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
        constexpr StrBuf(I i, char const* fstr = defaultFormatter<I>())
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

    template <typename Derived>
    class InterfaceLogger
    {
    public:
        /**
         * Function which performs logging with the `LOG` log level, only if `m_level` is at least `LOG`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        void log(std::string_view const& str, std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::LOG, str, loc);
        }

        /**
         * Function which performs logging with the `ERROR` log level, only if `m_level` is at least `LOG`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        void error(std::string_view const& str, std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::ERR, str, loc);
        }

        /**
         * Function which performs logging with the `ERROR` log level, only if `m_level` is at least `WARN`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        void warn(std::string_view const& str, std::source_location const& loc = std::source_location::current())
        {
            static_cast<Derived*>(this)->write(ELogLevel::WARNING, str, loc);
        }

        /**
         * Function which performs logging with the `ERROR` log level, only if `m_level` is at least `TRACE`
         * @param str the string to print
         * @param loc source location of the caller, auto calculated
         */
        void trace(std::string_view const& str, std::source_location const& loc = std::source_location::current())
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
        void log(std::string_view const&              str,
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
        void error(std::string_view const&              str,
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
        void warn(std::string_view const&              str,
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
        void trace(std::string_view const&              str,
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
        [[nodiscard]] bool enabled(ELogLevel level) const { return m_level <= level; }

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

    /**
     * Class implementing basic console logging while making use of the async IO facilities of the Windows and Linux
     * Operating system.
     * Note: OS API usage is not beneficial to performance here, but it's for learning and reference
     */
    class DMT_PLATFORM_API ConsoleLogger : public BaseLogger<ConsoleLogger>
    {
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
        explicit ConsoleLogger(ELogLevel level) : BaseLogger<ConsoleLogger>(level) {}

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
        void setLevel(ELogLevel level) { logger.setLevel(level); }

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str string to output
         * @param loc location of the log
         */
        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
        {
            logger.write(level, str, loc);
        }

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
                   std::source_location const&          loc)
        {
            logger.write(level, str, list, loc);
        }

        /**
         * CRTP overridden function to check if the true underlying logger is enabled on the log level
         * @param level log level requested
         * @return bool signaling whether the requested log level is enabled
         */
        bool enabled(ELogLevel level) const { return logger.enabled(level); }

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
} // namespace dmt
