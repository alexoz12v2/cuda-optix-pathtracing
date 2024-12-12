/**
 * @file platform-logging.cppm
 * @brief partition interface unit for platform module implementing
 * basic logging functionality. Key features:
 * - logging is performed on 3 statically allocated buffers (stderr, stdout, warning)
 * - such buffers are "leaky", ie they overwrite previous content if they are filled up
 * - logging macros/functions specify what to put on a buffer and then specify a destination, 2 types
 *   - console
 *   - window panel
 *
 * A possibility for the logger to console would be to be asynchronous
 *
 * Desired Usage:
 * - there will be 2 LogDisplays defined: Console Output and Window Panel output
 * - there should be a Macro which decides whether console output logs are turned off or not (compile time log level)
 * - there should be runtime functions to check whether a given log level is enabled or not (runtime log level)
 * @defgroup platform platform Module
 * @{
 */
module;

#include <concepts>
#include <format>
#include <string>
#include <string_view>
#include <utility>

#include <cassert>
#include <compare>
#include <cstdint>
#include <cstring>

/**
 * @brief module partition `platform:logging`
 */
export module platform:logging;

export namespace dmt
{

/**
 * @enum dmtELogLevel
 * @brief log levels for logger configuration
 */
enum class ELogLevel : uint8_t
{
    TRACE = 0, /** <Debug log level> */
    LOG   = 1, /** <Info log level> */
    WARN  = 2, /** <Warning log level> */
    ERROR = 3, /** <Error log level> */
    NONE  = 4, /** <Log disabled> */
};

constexpr uint8_t toUnderlying(ELogLevel level)
{
    return static_cast<uint8_t>(level);
}

// Spaceship operator (three-way comparison)
constexpr std::strong_ordering operator<=>(ELogLevel lhs, ELogLevel rhs) noexcept
{
    return toUnderlying(lhs) <=> toUnderlying(rhs);
}

namespace logcolor
{
inline constexpr std::string_view const reset        = "\033[0m";
inline constexpr std::string_view const red          = "\033[31m";
inline constexpr std::string_view const green        = "\033[32m";
inline constexpr std::string_view const brightYellow = "\033[93m";       // Bright yellow
inline constexpr std::string_view const greyGreen    = "\033[38;5;102m"; // Pale greenish
inline constexpr std::string_view const blue         = "\033[34m";
inline constexpr std::string_view const magenta      = "\033[35m";
inline constexpr std::string_view const cyan         = "\033[36m";
inline constexpr std::string_view const brightWhite  = "\033[97m";

inline constexpr std::string_view levelToColor(ELogLevel level)
{
    assert(level != ELogLevel::NONE);
    constexpr std::array<std::string_view, toUnderlying(ELogLevel::NONE)> colors{greyGreen, brightWhite, brightYellow, red};
    return colors[toUnderlying(level)];
}
} // namespace logcolor

enum class ELogDisplay : uint8_t
{
    Console,
    WindowPanel,
};

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
    constexpr StrBuf(char const* str, int32_t len) : str(str), len(len)
    {
    }

    StrBuf(char const* str) : str(str), len(static_cast<int32_t>(std::strlen(str)))
    {
    }

    /**
     * Converting constructor from a string_view. NOT `explicit` on purpose
     * @param view
     */
    constexpr StrBuf(std::string_view const& view) : str(view.data()), len(static_cast<int32_t>(view.length()))
    {
    }

    /**
     * Converting constructor for formatting booleans
     * @param b
     */
    constexpr StrBuf(bool b) : StrBuf(b ? m_strue : m_sfalse)
    {
    }

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
     * Constructor from an integral value. If the format string is reasonable, it shouldn't allocate
     * @tparam I
     * @param i
     * @param fstr
     */
    template <std::integral I>
    constexpr StrBuf(I i, char const* fstr = "%d")
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
    static inline constexpr std::string_view m_strue  = "true";
    static inline constexpr std::string_view m_sfalse = "false";

    template <typename T>
        requires(std::integral<T> || std::floating_point<T>)
    constexpr void initialize(T value, char const* fstr)
    {
        // SBO optimized string
        std::string s(15, '\0');
        auto        written = std::snprintf(&s[0], s.size(), fstr, value);
        assert(written > 0 && "invalid formatting string");
        assert(s.capacity() == 15 && "String was allocated");
        s.resize(static_cast<uint32_t>(written));
        auto sz = static_cast<int32_t>(s.size());
        std::memcpy(buf, s.data(), static_cast<uint32_t>(sz));
        buf[sz] = '\0';
        len     = -sz;
    }
};

/**
 * Class which formats all the given arguments into a local buffer
 */
class CircularOStringStream
{
public:
    static constexpr uint32_t bufferSize = 4096;
    CircularOStringStream();
    CircularOStringStream& operator<<(StrBuf const& buf);
    CircularOStringStream& operator<<(char const c);
    void                   clear();
    void                   logInitList(char const* formatStr, std::initializer_list<StrBuf> const& args);
    std::string_view       str();

private:
    char     m_buffer[bufferSize]{};
    uint32_t m_pos{0};
};

// clang-format off
template <typename T>
concept LogDisplay = requires(T t)
{
    typename T::Traits;
    requires std::is_same_v<std::remove_cvref_t<decltype(T::Traits::displayType)>, ELogDisplay>;
    requires requires (ELogLevel level, std::string_view str)
    {
        {t.setLevel(level)} -> std::same_as<void>;
        {t.enabled(level)} -> std::same_as<bool>;
        {t.logEnabled()} -> std::same_as<bool>;
        {t.errorEnabled()} -> std::same_as<bool>;
        {t.traceEnabled()} -> std::same_as<bool>;
        {t.warnEnabled()} -> std::same_as<bool>;
        {t.write(level, str)} -> std::same_as<void>;
        {t.log(str)} -> std::same_as<void>;
        {t.error(str)} -> std::same_as<void>;
        {t.warn(str)} -> std::same_as<void>;
        {t.trace(str)} -> std::same_as<void>;
        requires requires(std::initializer_list<StrBuf> const &list)
        {
            {t.write(level, str, list)} -> std::same_as<void>;
            {t.log(str, list)} -> std::same_as<void>;
            {t.error(str, list)} -> std::same_as<void>;
            {t.warn(str, list)} -> std::same_as<void>;
            {t.trace(str, list)} -> std::same_as<void>;
        };
    };
};
// clang-format on

template <typename Derived>
class BaseLogger
{
public:
    explicit BaseLogger(ELogLevel level = ELogLevel::LOG) : m_level(level){};

    void setLevel(ELogLevel level)
    {
        m_level = level;
    }

    [[nodiscard]] bool enabled(ELogLevel level) const
    {
        return m_level <= level;
    }

    [[nodiscard]] bool logEnabled() const
    {
        return enabled(ELogLevel::LOG);
    }

    [[nodiscard]] bool errorEnabled() const
    {
        return enabled(ELogLevel::ERROR);
    }

    [[nodiscard]] bool traceEnabled() const
    {
        return enabled(ELogLevel::TRACE);
    }

    [[nodiscard]] bool warnEnabled() const
    {
        return enabled(ELogLevel::WARN);
    }

    void log(std::string_view const& str)
    {
        static_cast<Derived*>(this)->write(ELogLevel::LOG, str);
    }
    void error(std::string_view const& str)
    {
        static_cast<Derived*>(this)->write(ELogLevel::ERROR, str);
    }
    void warn(std::string_view const& str)
    {
        static_cast<Derived*>(this)->write(ELogLevel::WARN, str);
    }
    void trace(std::string_view const& str)
    {
        static_cast<Derived*>(this)->write(ELogLevel::TRACE, str);
    }

    void log(std::string_view const& str, std::initializer_list<StrBuf> const& list)
    {
        static_cast<Derived*>(this)->write(ELogLevel::LOG, str, list);
    }
    void error(std::string_view const& str, std::initializer_list<StrBuf> const& list)
    {
        static_cast<Derived*>(this)->write(ELogLevel::ERROR, str, list);
    }
    void warn(std::string_view const& str, std::initializer_list<StrBuf> const& list)
    {
        static_cast<Derived*>(this)->write(ELogLevel::WARN, str, list);
    }
    void trace(std::string_view const& str, std::initializer_list<StrBuf> const& list)
    {
        static_cast<Derived*>(this)->write(ELogLevel::TRACE, str, list);
    }

protected:
    ELogLevel m_level;
};

class ConsoleLogger : public BaseLogger<ConsoleLogger>
{
public:
    ConsoleLogger(std::string_view const& prefix, ELogLevel level = ELogLevel::LOG);

    struct Traits
    {
        static constexpr ELogDisplay displayType = ELogDisplay::Console;
    };

    void write(ELogLevel level, std::string_view const& str);

    void write(ELogLevel level, std::string_view const& str, std::initializer_list<StrBuf> const& list);

private:
    CircularOStringStream m_oss;
    // TODO
    std::string_view m_prefix = "";
};

} // namespace dmt

/** @} */