#ifndef DMT_TRAINING_COOPERATIVE_GROUPS_PARSING_UTILS_H
#define DMT_TRAINING_COOPERATIVE_GROUPS_PARSING_UTILS_H

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <cwctype> // iswspace
#include <cctype>  // isspace
#include <system_error>

namespace dmt {
    template <typename T>
    inline constexpr bool is_bool_v = std::is_same_v<std::remove_cv_t<T>, bool>;

    // core implementation for char and wchar_t
    template <typename T, typename CharT>
    std::optional<T> parse_number_impl(std::basic_string_view<CharT> sv)
    {
        static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
        static_assert(!is_bool_v<T>, "bool is not supported");

        if (sv.empty())
            return std::nullopt;

        // disallow leading or trailing whitespace to match strict parsing semantics
        if constexpr (std::is_same_v<CharT, char>)
        {
            if (std::isspace(static_cast<unsigned char>(sv.front())) || std::isspace(static_cast<unsigned char>(sv.back())))
                return std::nullopt;
        }
        else
        { // wchar_t
            if (std::iswspace(sv.front()) || std::iswspace(sv.back()))
                return std::nullopt;
        }

        if constexpr (std::is_integral_v<T>)
        {
            // use from_chars for narrow char
            if constexpr (std::is_same_v<CharT, char>)
            {
                T    value{};
                auto first     = sv.data();
                auto last      = sv.data() + sv.size();
                auto [ptr, ec] = std::from_chars(first, last, value, 10);
                if (ec != std::errc{} || ptr != last)
                    return std::nullopt;
                return value;
            }
            else if constexpr (std::is_same_v<CharT, wchar_t>)
            {
                // wide-char path using wcstoll / wcstoull
                wchar_t const* begin  = sv.data();
                wchar_t*       endptr = nullptr;
                errno                 = 0;

                if constexpr (std::is_signed_v<T>)
                {
                    long long tmp = std::wcstoll(begin, &endptr, 10);
                    if (endptr != begin + sv.size())
                        return std::nullopt; // not fully consumed
                    if (errno == ERANGE)
                        return std::nullopt;
                    if (tmp < std::numeric_limits<T>::lowest() || tmp > std::numeric_limits<T>::max())
                        return std::nullopt;
                    return static_cast<T>(tmp);
                }
                else
                { // unsigned
                    unsigned long long const tmp = std::wcstoull(begin, &endptr, 10);
                    if (endptr != begin + sv.size())
                        return std::nullopt;
                    if (errno == ERANGE)
                        return std::nullopt;
                    if (tmp > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
                        return std::nullopt;
                    return static_cast<T>(tmp);
                }
            }
            else
            {
                static_assert(std::is_same_v<CharT, CharT>, "Unsupported character type");
            }
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                T    value{};
                auto first = sv.data();
                auto last  = sv.data() + sv.size();
                // floating-point from_chars is available in modern standard libraries
                auto [ptr, ec] = std::from_chars(first, last, value);
                if (ec != std::errc{} || ptr != last)
                    return std::nullopt;
                return value;
            }
            else if constexpr (std::is_same_v<CharT, wchar_t>)
            {
                wchar_t const* begin  = sv.data();
                wchar_t*       endptr = nullptr;
                errno                 = 0;
                if constexpr (std::is_same_v<T, float>)
                {
                    // wcstof might not be standard everywhere; fallback to wcstod then check range
                    double const tmp = std::wcstod(begin, &endptr);
                    if (endptr != begin + sv.size())
                        return std::nullopt;
                    if (errno == ERANGE)
                        return std::nullopt;
                    if (tmp < std::numeric_limits<float>::lowest() || tmp > std::numeric_limits<float>::max())
                        return std::nullopt;
                    return static_cast<float>(tmp);
                }
                else if constexpr (std::is_same_v<T, double>)
                {
                    double const tmp = std::wcstod(begin, &endptr);
                    if (endptr != begin + sv.size())
                        return std::nullopt;
                    if (errno == ERANGE)
                        return std::nullopt;
                    return static_cast<double>(tmp);
                }
                else
                { // long double
                    long double const tmp = std::wcstold(begin, &endptr);
                    if (endptr != begin + sv.size())
                        return std::nullopt;
                    if (errno == ERANGE)
                        return std::nullopt;
                    return static_cast<long double>(tmp);
                }
            }
            else
            {
                static_assert(std::is_same_v<CharT, CharT>, "Unsupported character type");
            }
        }
        else
        {
            return std::nullopt; // unreachable due to static_assert earlier
        }

        return std::nullopt;
    }

    // convenience overloads for common string types
    template <typename T>
    std::optional<T> parse_number(std::string_view sv)
    {
        return parse_number_impl<T, char>(sv);
    }

    template <typename T>
    std::optional<T> parse_number(std::string const& s)
    {
        return parse_number_impl<T, char>(std::string_view{s});
    }

    template <typename T>
    std::optional<T> parse_number(std::wstring_view sv)
    {
        return parse_number_impl<T, wchar_t>(sv);
    }

    template <typename T>
    std::optional<T> parse_number(std::wstring const& s)
    {
        return parse_number_impl<T, wchar_t>(std::wstring_view{s});
    }

    // also convenience for C-strings
    template <typename T>
    std::optional<T> parse_number(char const* s)
    {
        return parse_number<T>(std::string_view{s});
    }

    template <typename T>
    std::optional<T> parse_number(wchar_t const* s)
    {
        return parse_number<T>(std::wstring_view{s});
    }

    size_t prettyPrintFloatMaxWidth(float const* data, size_t count);

} // namespace dmt

#endif // DMT_TRAINING_COOPERATIVE_GROUPS_PARSING_UTILS_H
