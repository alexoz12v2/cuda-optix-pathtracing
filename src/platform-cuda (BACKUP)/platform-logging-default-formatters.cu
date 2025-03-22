#include "platform-logging-default-formatters.h"

#include <limits>
#include <utility>

#include <cassert>
#include <cstdint>

// Source: https://github.com/eyalroz/cuda-kat/blob/development/src/kat/on_device/c_standard_library/printf.cu#L345
namespace dmt::detail {
    /**
     * max_chars is 0
     * buffer is non-null
     * function is non-null
     * one of these three must hold
     */
    struct output_gadget_t
    {
        __device__ output_gadget_t(char* _buffer, int32_t _bufferSize) :
        buffer(_buffer),
        maxChars(static_cast<uint32_t>(_bufferSize > 0 ? _bufferSize : 0))
        {
        }

        void (*function)(char _c, void* _extraArg) = nullptr;
        void*    extraArg                          = nullptr;
        char*    buffer                            = nullptr;
        uint32_t pos                               = 0;
        uint32_t maxChars                          = 0;
    };

    enum class EFlags : uint32_t
    {
        none      = 0,
        zeropad   = 1U << 0U,
        left      = 1U << 1U,
        plus      = 1U << 2U,
        space     = 1U << 3U,
        hash      = 1U << 4U,
        uppercase = 1U << 5U,
        char_     = 1U << 6U,
        short_    = 1U << 7U,
        int_      = 1U << 8U,
        long_     = 1U << 9U,
        long_long = 1U << 10U,
        precision = 1U << 11U,
        adapt_exp = 1U << 12U,
        pointer   = 1U << 13U, // Note: Similar, but not identical, effect as hash
        signed_   = 1U << 14U,
        int8      = char_,
        int16     = short_,
        int32     = int_,
        int64     = long_
    };

    enum class EBase : uint32_t
    {
        binary  = 2,
        octal   = 8,
        decimal = 10,
        hex     = 16
    };

    // Enable bitwise operators for EFlags
    constexpr EFlags operator|(EFlags lhs, EFlags rhs) noexcept
    {
        using underlying = std::underlying_type_t<EFlags>;
        return static_cast<EFlags>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
    }

    constexpr EFlags operator&(EFlags lhs, EFlags rhs) noexcept
    {
        using underlying = std::underlying_type_t<EFlags>;
        return static_cast<EFlags>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
    }

    constexpr EFlags operator^(EFlags lhs, EFlags rhs) noexcept
    {
        using underlying = std::underlying_type_t<EFlags>;
        return static_cast<EFlags>(static_cast<underlying>(lhs) ^ static_cast<underlying>(rhs));
    }

    constexpr EFlags operator~(EFlags flag) noexcept
    {
        using underlying = std::underlying_type_t<EFlags>;
        return static_cast<EFlags>(~static_cast<underlying>(flag));
    }

    static constexpr bool any(EFlags flags) noexcept { return flags != EFlags::none; }

    constexpr EFlags& operator|=(EFlags& lhs, EFlags rhs) noexcept
    {
        lhs = lhs | rhs;
        return lhs;
    }

    constexpr EFlags& operator&=(EFlags& lhs, EFlags rhs) noexcept
    {
        lhs = lhs & rhs;
        return lhs;
    }

    constexpr EFlags& operator^=(EFlags& lhs, EFlags rhs) noexcept
    {
        lhs = lhs ^ rhs;
        return lhs;
    }

    static __device__ EFlags parseFlags(char const** format)
    {
        EFlags flags = EFlags::none;
        do
        {
            switch (**format)
            {
                case '0':
                    flags |= EFlags::zeropad;
                    (*format)++;
                    break;
                case '-':
                    flags |= EFlags::left;
                    (*format)++;
                    break;
                case '+':
                    flags |= EFlags::plus;
                    (*format)++;
                    break;
                case ' ':
                    flags |= EFlags::space;
                    (*format)++;
                    break;
                case '#':
                    flags |= EFlags::hash;
                    (*format)++;
                    break;
                default: return flags;
            }
        } while (true);
    }

    static __device__ void putchar_via_gadget(output_gadget_t& gadget, char c)
    {
        uint32_t write_pos = gadget.pos++;
        // We're _always_ increasing pos, so as to count how may characters
        // _would_ have been written if not for the max_chars limitation
        if (write_pos >= gadget.maxChars)
        {
            return;
        }
        if (gadget.function != nullptr)
        {
            // No check for c == '\0' .
            gadget.function(c, gadget.extraArg);
        }
        else
        {
            // it must be the case that gadget->buffer != nullptr , due to the constraint
            // on output_gadget_t ; and note we're relying on write_pos being non-negative.
            gadget.buffer[write_pos] = c;
        }
    }

    static constexpr __device__ bool is_digit_(char ch) { return (ch >= '0') && (ch <= '9'); }


    // internal ASCII string to printf_size_t conversion
    static __device__ uint32_t atou_(char const** str)
    {
        uint32_t i = 0U;
        while (is_digit_(**str))
            i = i * 10U + static_cast<uint32_t>(*((*str)++) - '0');
        return i;
    }

    template <std::integral I>
    static constexpr __device__ uint64_t abs_for_printing(I i)
    {
        return i > 0 ? static_cast<uint64_t>(i) : static_cast<uint64_t>(-i);
    }

    static __device__ void out_rev_(output_gadget_t& output, char const* DMT_RESTRICT buf, uint32_t len, uint32_t width, EFlags flags)
    {
        uint32_t const start = output.pos;
        // pad spaces up to `width`
        if (any(flags & EFlags::left) && any(flags & EFlags::zeropad))
            for (uint32_t i = 0; i < width; ++i)
                putchar_via_gadget(output, ' ');

        // reverse string
        while (len)
            putchar_via_gadget(output, buf[--len]);

        // append spaces up to given width
        if ((flags & EFlags::left) != EFlags::none)
            while (output.pos - start < width)
                putchar_via_gadget(output, ' ');
    }

    static __device__ void print_integer_finalization(
        output_gadget_t&   output,
        char* DMT_RESTRICT buf,
        uint32_t           len,
        bool               negative,
        EBase              base,
        uint32_t           precision,
        uint32_t           width,
        EFlags             flags)
    {
        uint32_t unpaddedLen = len;
        // pad with leading zeros
        {
            if ((flags & EFlags::left) == EFlags::none)
            {
                if (width && (flags & EFlags::zeropad) != EFlags::none &&
                    ((flags & (EFlags::plus | EFlags::space)) != EFlags::none))
                    --width;
                while ((flags & EFlags::zeropad) != EFlags::none && (len < width) && (len < 32))
                    buf[len++] = '0';
            }

            while (len < precision && len < 32)
                buf[len++] = '0';

            if (base == EBase::octal && len > unpaddedLen)
                flags &= ~EFlags::hash;
        }

        // handle hash
        if ((flags & (EFlags::hash | EFlags::pointer)) != EFlags::none)
        {
            if ((flags & EFlags::precision) == EFlags::none && len && (len == precision || len == width))
            {
                if (unpaddedLen < len)
                    --len;
                if (len && (base == EBase::hex || base == EBase::binary) && unpaddedLen < len)
                    --len; // extra for 0x or 0b
            }
            if (base == EBase::hex && (flags & EFlags::uppercase) == EFlags::none && len < 32)
                buf[len++] = 'x';
            else if (base == EBase::hex && (flags & EFlags::uppercase) != EFlags::none && len < 32)
                buf[len++] = 'X';
            else if (base == EBase::binary && len < 32)
                buf[len++] = 'b';

            if (len < 32)
                buf[len++] = '0';
        }

        if (len < 32)
        {
            if (negative)
                buf[len++] = '-';
            else if ((flags & EFlags::plus) != EFlags::none)
                buf[len++] = '+';
            else if ((flags & EFlags::space) != EFlags::none)
                buf[len++] = ' ';
        }

        out_rev_(output, buf, len, width, flags);
    }

    static __device__ void print_integer(
        output_gadget_t& output,
        uint64_t         unsignedValue,
        bool             negative,
        EBase            base,
        uint32_t         precision,
        uint32_t         width,
        EFlags           flags)
    {
        char     buf[32];
        uint32_t len = 0u;
        if (unsignedValue == 0)
        {
            if ((flags & EFlags::precision) == EFlags::none)
            {
                buf[len++] = '0';
                flags &= ~EFlags::hash;
            }
            else if (base == EBase::hex)
                flags &= ~EFlags::hash;
        }
        else
        {
            do
            {
                char const digit = static_cast<char>(unsignedValue % static_cast<std::underlying_type_t<EBase>>(base));
                buf[len++]       = digit < 10 ? '0' + digit
                                              : ((flags & EFlags::uppercase) != EFlags::none ? 'A' : 'a') + (digit - 10);
                unsignedValue /= static_cast<std::underlying_type_t<EBase>>(base);
            } while (unsignedValue != 0 && len < 32);
        }

        print_integer_finalization(output, buf, len, negative, base, precision, width, flags);
    }

    namespace f64 {
        static constexpr uint32_t numBits            = 64;
        static constexpr uint32_t baseExponent       = 1023;
        static constexpr uint32_t storedMantissaBits = 52;
        static constexpr uint32_t exponentMask       = 0x7FFU;
        static constexpr uint32_t decimalBufferSize  = 32;

        union Double
        {
            uint64_t u;
            double   f;
        };

        static constexpr __device__ Double wrap(double x)
        {
            Double ret;
            ret.f = x;
            return ret;
        }

        static constexpr __device__ int32_t exp2(Double d)
        {
            return static_cast<int32_t>(((d.u >> storedMantissaBits) & exponentMask) - baseExponent);
        }

        struct Components
        {
            int64_t integral;
            int64_t fractional;
            bool    isNegative;
        };

        static constexpr __device__ int32_t get_sign_bit(double x)
        {
            return static_cast<int32_t>(wrap(x).u >> (numBits - 1));
        }

        static constexpr __device__ int32_t get_exp2(double x) { return exp2(wrap(x)); }

        static __device__ double power_of_10(int e)
        {
            static __constant__ double const powers_of_10[18] =
                {1e00, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06, 1e07, 1e08, 1e09, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17};
            return (e >= 0 && e <= 17) ? powers_of_10[e] : 1.0;
        }

        static __device__ Components getComponents(double number, uint32_t precision)
        {
            Components resComps;
            resComps.isNegative = get_sign_bit(number);
            double absNumber    = resComps.isNegative ? -number : number;
            resComps.integral   = static_cast<int64_t>(absNumber);
            double remainder    = (absNumber - static_cast<double>(resComps.integral)) *
                               power_of_10(static_cast<int32_t>(precision));
            resComps.fractional = static_cast<int64_t>(remainder);
            remainder -= static_cast<double>(resComps.fractional);

            if (remainder > 0.5)
            {
                ++resComps.fractional;
                if (static_cast<double>(resComps.fractional) >= power_of_10(static_cast<int32_t>(precision)))
                { // rollover: 0.99 with precision 1 is 1.0
                    resComps.fractional = 0;
                    ++resComps.integral;
                }
            }
            else if (remainder == 0.5 && (resComps.fractional == 0U || (resComps.fractional & 1U)))
                ++resComps.fractional;

            if (precision == 0U) // print only first digit, integral
            {
                remainder             = absNumber - static_cast<double>(resComps.integral);
                bool const notHalfway = (!(remainder < 0.5) || (remainder > 0.5));
                if (notHalfway && (resComps.integral & 1))
                    ++resComps.integral;
            }

            return resComps;
        }

        struct ScalingFactor
        {
            double rawFactor;
            bool   multiply; // divide if false
        };

        static __device__ double applyScaling(double num, ScalingFactor normalization)
        {
            return normalization.multiply ? num * normalization.rawFactor : num / normalization.rawFactor;
        }

        static __device__ double unapplyScaling(double num, ScalingFactor normalization)
        {
            return normalization.multiply ? num / normalization.rawFactor : num * normalization.rawFactor;
        }

        static __device__ ScalingFactor updateNormalizetion(ScalingFactor sf, double extra)
        {
            ScalingFactor result;
            int32_t const factorExp2 = get_exp2(sf.rawFactor);
            int32_t const extraExp2  = get_exp2(extra);
            if (abs(factorExp2) > abs(extraExp2))
            {
                result.multiply  = false;
                result.rawFactor = sf.rawFactor / extra;
            }
            else
            {
                result.multiply  = true;
                result.rawFactor = extra / sf.rawFactor;
            }

            return result;
        }

        static __device__ Components getNormalizedComponents(
            bool          negative,
            uint32_t      precision,
            double        nonNormalized,
            ScalingFactor normalization,
            int           flooredExp10)
        {
            Components compos;
            compos.isNegative                        = negative;
            double     scaled                        = applyScaling(nonNormalized, normalization);
            bool const closeToRepresentationExtremum = (-flooredExp10 + static_cast<int32_t>(precision)) >=
                                                       std::numeric_limits<double>::max_exponent10 - 1;
            if (closeToRepresentationExtremum)
            {
                compos = getComponents(negative ? -scaled : scaled, precision);
            }
            else
            {
                compos.integral = static_cast<int64_t>(scaled);
                double const remainder = nonNormalized - unapplyScaling(static_cast<double>(compos.integral), normalization);
                double const        prec_power_of_10    = power_of_10(static_cast<int32_t>(precision));
                ScalingFactor const accountForPrecision = updateNormalizetion(normalization, prec_power_of_10);
                double              scaledRemainder     = applyScaling(remainder, accountForPrecision);
                double const        roundingThreshold   = 0.5;

                // round to nearest procedure
                compos.fractional = static_cast<int64_t>(scaledRemainder);
                scaledRemainder -= static_cast<double>(compos.fractional);
                compos.fractional += (scaledRemainder >= roundingThreshold);
                if (scaledRemainder == roundingThreshold) // banker's rouding = round towards even number
                    compos.fractional &= ~0x1LL;

                // handle rollover
                if (static_cast<double>(compos.fractional) >= prec_power_of_10)
                {
                    compos.fractional = 0;
                    ++compos.integral;
                }
            }

            return compos;
        }

        static __device__ void print_broken_up_decimal(
            Components         number,
            output_gadget_t&   output,
            uint32_t           precision,
            uint32_t           width,
            EFlags             flags,
            char* DMT_RESTRICT buf,
            uint32_t           len)
        {
            if (precision != 0U)
            {
                uint32_t count              = precision;
                bool     adaptiveAndNotHash = (flags & EFlags::adapt_exp) != EFlags::none &&
                                          (flags & EFlags::hash) == EFlags::none;
                // skip trailing 0 digits if %g or %G
                if (adaptiveAndNotHash && number.fractional > 0)
                {
                    while (true)
                    {
                        int64_t digit = number.fractional % 10ULL;
                        if (digit != 0)
                            break;
                        --count;
                        number.fractional /= 10ULL;
                    }
                }

                if (number.fractional > 0 || !adaptiveAndNotHash)
                {
                    while (len < decimalBufferSize)
                    {
                        --count;
                        buf[len++] = '0' + static_cast<char>(number.fractional % 10ULL);
                        if (!(number.fractional /= 10ULL))
                            break;
                    }
                    // add extra 0s
                    while (len < decimalBufferSize && count > 0U)
                    {
                        buf[len++] = '0';
                        --count;
                    }
                    if (len < decimalBufferSize)
                        buf[len++] = '.';
                }
            }
            else // precision == 0
            {
                if ((flags & EFlags::hash) != EFlags::none && len < decimalBufferSize)
                    buf[len++] = '.';
            }

            // now write integer part of the number (we write on buf reversed)
            while (len < decimalBufferSize)
            {
                buf[len++] = '0' + static_cast<char>(number.integral % 10ULL);
                if (!(number.integral /= 10ULL))
                    break;
            }

            // pad leading zeros and sign
            if ((flags & EFlags::left) == EFlags::none && (flags & EFlags::zeropad) != EFlags::none)
            {
                if (width && (number.isNegative || (flags & (EFlags::plus | EFlags::space)) != EFlags::none))
                    --width;
                while (len < width && len < decimalBufferSize)
                    buf[len++] = '0';
            }

            if (len < decimalBufferSize)
            {
                if (number.isNegative)
                    buf[len++] = '-';
                else if ((flags & EFlags::plus) != EFlags::none)
                    buf[len++] = '+';
                else if ((flags & EFlags::space) != EFlags::none)
                    buf[len++] = ' ';
            }

            out_rev_(output, buf, len, width, flags);
        }
    } // namespace f64

    static __device__ void print_exponential_number(
        output_gadget_t&   output,
        double const       value,
        uint32_t           precision,
        uint32_t           width,
        EFlags             flags,
        char* DMT_RESTRICT buf,
        uint32_t           len)
    {
        bool const         isNegative                        = f64::get_sign_bit(value);
        double             absNumber                         = isNegative ? -value : value;
        int32_t            flooredExp10                      = 1;
        bool               abs_exp10_covered_by_powers_table = false;
        f64::ScalingFactor normalization;
        if (absNumber == 0.0)
            flooredExp10 = 0;
        else
        {
            double exp10                      = log10(absNumber);
            flooredExp10                      = floor(exp10);
            double p10                        = pow(10, flooredExp10);
            normalization.rawFactor           = p10;
            abs_exp10_covered_by_powers_table = false;
        }

        bool fallBackToDecimalOnlyMode = false;
        if ((flags & EFlags::adapt_exp) != EFlags::none)
        {
            int requiredSignificantDigits = precision == 0 ? 1 : static_cast<int32_t>(precision);
            fallBackToDecimalOnlyMode     = flooredExp10 >= -4 && flooredExp10 < requiredSignificantDigits;
            int32_t adjPrecision = fallBackToDecimalOnlyMode ? static_cast<int32_t>(precision) - 1 - flooredExp10
                                                             : static_cast<int32_t>(precision) - 1;
            precision            = adjPrecision > 0 ? static_cast<uint32_t>(adjPrecision) : 0U;
            flags |= EFlags::precision;
        }

        normalization.multiply                  = flooredExp10 < 0 && abs_exp10_covered_by_powers_table;
        bool            shouldSkipNormalization = fallBackToDecimalOnlyMode || flooredExp10 == 0;
        f64::Components decimalPart             = shouldSkipNormalization
                                                      ? f64::getComponents(value, precision)
                                                      : f64::getNormalizedComponents(isNegative, precision, absNumber, normalization, flooredExp10);

        if (fallBackToDecimalOnlyMode)
        { // special case: rollover
            if ((flags & EFlags::adapt_exp) != EFlags::none && flooredExp10 >= -1 &&
                decimalPart.integral == f64::power_of_10(flooredExp10 + 1))
            {
                ++flooredExp10;
                --precision;
                assert(decimalPart.fractional == 0LL);
            }
        }
        else
        {
            if (decimalPart.integral >= 10)
            {
                ++flooredExp10;
                decimalPart.integral   = 1;
                decimalPart.fractional = 0;
            }
        }

        uint32_t const exp10_part_width   = fallBackToDecimalOnlyMode ? 0U : ((abs(flooredExp10) < 100) ? 4U : 5U);
        uint32_t const decimal_part_width = ((flags & EFlags::left) != EFlags::none && exp10_part_width)
                                                ? 0U // padding on the right, then we append padding later
                                                : ((width > exp10_part_width) // padding on the left. do we fit in the width?
                                                       ? width - exp10_part_width
                                                       : 0U);
        uint32_t const printed_exponent_start_pos = output.pos;
        f64::print_broken_up_decimal(decimalPart, output, precision, decimal_part_width, flags, buf, len);

        if (!fallBackToDecimalOnlyMode)
        {
            putchar_via_gadget(output, (flags & EFlags::uppercase) != EFlags::none ? 'E' : 'e');
            print_integer(output,
                          abs_for_printing(flooredExp10),
                          flooredExp10 < 0,
                          EBase::decimal,
                          0,
                          exp10_part_width - 1,
                          EFlags::zeropad | EFlags::plus);
            if ((flags & EFlags::left) != EFlags::none)
            { // right pad
                while (output.pos - printed_exponent_start_pos < width)
                    putchar_via_gadget(output, ' ');
            }
        }
    }

    static __device__ void print_decimal_number(
        output_gadget_t&   output,
        double             value,
        uint32_t           precision,
        uint32_t           width,
        EFlags             flags,
        char* DMT_RESTRICT buf,
        uint32_t           len)
    {
        f64::Components compos = f64::getComponents(value, precision);
        f64::print_broken_up_decimal(compos, output, precision, width, flags, buf, len);
    }

    static __device__ void print_floating_point(
        output_gadget_t& output,
        double           value,
        uint32_t         precision,
        uint32_t         width,
        EFlags           flags,
        bool             preferExponential)
    {
        static constexpr double   float_notation_threshold      = 1e9; // beyond this, force exponential notation
        static constexpr uint32_t default_precision             = 6;
        static constexpr uint32_t num_decimal_digits_in_int64_t = 18;
        static constexpr uint32_t max_precision                 = num_decimal_digits_in_int64_t - 1;
        static constexpr uint32_t decimal_buffer_size           = 32;
        char                      buf[decimal_buffer_size];
        uint32_t                  len = 0U;

        if (value != value)
            out_rev_(output, "nan", 3, width, flags);
        else if (value < -std::numeric_limits<double>::max())
            out_rev_(output, "-inf", 4, width, flags);
        else if (value > std::numeric_limits<double>::max())
            out_rev_(output,
                     (flags & EFlags::plus) != EFlags::none ? "+inf" : "inf",
                     (flags & EFlags::plus) != EFlags::none ? 4 : 3,
                     width,
                     flags);
        else if (!preferExponential && value > float_notation_threshold && value < -float_notation_threshold)
            print_exponential_number(output, value, precision, width, flags, buf, len);
        else
        {
            if ((flags & EFlags::precision) == EFlags::none)
                precision = default_precision;
            while (len < decimal_buffer_size && precision > max_precision)
            {
                buf[len++] = '0';
                --precision;
            }

            if (preferExponential)
                print_exponential_number(output, value, precision, width, flags, buf, len);
            else
                print_decimal_number(output, value, precision, width, flags, buf, len);
        }
    }

    static __device__ uint32_t strnlen_s_(char const* str, uint32_t maxSz)
    {
        char const* s = str;
        for (/**/; *s && maxSz--; ++s)
            ;
        return static_cast<uint32_t>(s - str);
    }

    static __device__ void append_termination_with_gadget(output_gadget_t& gadget)
    {
        if (gadget.function != nullptr || gadget.maxChars == 0)
            return;
        if (gadget.buffer == nullptr)
            return;
        uint32_t nullPos       = gadget.pos < gadget.maxChars ? gadget.pos : gadget.maxChars - 1;
        gadget.buffer[nullPos] = '\0';
    }

    static __device__ int vsnprintf(output_gadget_t& output, char const* format, va_list args)
    {
        // constants
        static constexpr bool     prefer_decimal           = false;
        static constexpr bool     prefer_exponential       = true;
        static constexpr uint32_t max_possible_buffer_size = 32;

        while (*format)
        {
            // format specifier?  %[flags][width][.precision][length]
            if (*format != '%')
            {
                // no
                putchar_via_gadget(output, *format);
                format++;
                continue;
            }
            else
            {
                // yes, evaluate it
                format++;
            }

            EFlags flags = parseFlags(&format);

            // evaluate width field
            uint32_t width = 0U;
            if (is_digit_(*format))
            {
                width = static_cast<uint32_t>(atou_(&format));
            }
            else if (*format == '*')
            {
                int const w = va_arg(args, int);
                if (w < 0)
                {
                    flags |= EFlags::left; // reverse padding
                    width = static_cast<uint32_t>(-w);
                }
                else
                {
                    width = static_cast<uint32_t>(w);
                }
                format++;
            }

            // evaluate precision field
            uint32_t precision = 0U;
            if (*format == '.')
            {
                flags |= EFlags::precision;
                format++;
                if (is_digit_(*format))
                {
                    precision = static_cast<uint32_t>(atou_(&format));
                }
                else if (*format == '*')
                {
                    int const precision_ = va_arg(args, int);
                    precision            = precision_ > 0 ? static_cast<uint32_t>(precision_) : 0U;
                    format++;
                }
            }

            // evaluate length field
            switch (*format)
            {
                case 'I':
                {
                    format++;
                    // Greedily parse for size in bits: 8, 16, 32 or 64
                    switch (*format)
                    {
                        case '8':
                            flags |= EFlags::int8;
                            format++;
                            break;
                        case '1':
                            format++;
                            if (*format == '6')
                            {
                                format++;
                                flags |= EFlags::int16;
                            }
                            break;
                        case '3':
                            format++;
                            if (*format == '2')
                            {
                                format++;
                                flags |= EFlags::int32;
                            }
                            break;
                        case '6':
                            format++;
                            if (*format == '4')
                            {
                                format++;
                                flags |= EFlags::int64;
                            }
                            break;
                        default: break;
                    }
                    break;
                }
                case 'l':
                    flags |= EFlags::long_;
                    format++;
                    if (*format == 'l')
                    {
                        flags |= EFlags::long_long;
                        format++;
                    }
                    break;
                case 'h':
                    flags |= EFlags::short_;
                    format++;
                    if (*format == 'h')
                    {
                        flags |= EFlags::char_;
                        format++;
                    }
                    break;
                case 't':
                case 'j':
                case 'z':
                    flags |= EFlags::long_;
                    format++;
                    break;
                default: break;
            }

            // evaluate specifier
            switch (*format)
            {
                case 'd':
                case 'i':
                case 'u':
                case 'x':
                case 'X':
                case 'o':
                case 'b':
                {

                    if (*format == 'd' || *format == 'i')
                    {
                        flags |= EFlags::signed_;
                    }

                    EBase base;
                    if (*format == 'x' || *format == 'X')
                    {
                        base = EBase::hex;
                    }
                    else if (*format == 'o')
                    {
                        base = EBase::octal;
                    }
                    else if (*format == 'b')
                    {
                        base = EBase::binary;
                    }
                    else
                    {
                        base = EBase::decimal;
                        flags &= ~EFlags::hash; // decimal integers have no alternative presentation
                    }

                    if (*format == 'X')
                    {
                        flags |= EFlags::uppercase;
                    }

                    format++;
                    // ignore '0' flag when precision is given
                    if ((flags & EFlags::precision) != EFlags::none)
                    {
                        flags &= ~EFlags::zeropad;
                    }

                    if ((flags & EFlags::signed_) != EFlags::none)
                    {
                        // A signed specifier: d, i or possibly I + bit size if enabled

                        if ((flags & EFlags::long_long) != EFlags::none)
                        {
                            long long const value = va_arg(args, long long);
                            print_integer(output, abs_for_printing(value), value < 0, base, precision, width, flags);
                        }
                        else if ((flags & EFlags::long_) != EFlags::none)
                        {
                            long const value = va_arg(args, long);
                            print_integer(output, abs_for_printing(value), value < 0, base, precision, width, flags);
                        }
                        else
                        {
                            // We never try to interpret the argument as something potentially-smaller than int,
                            // due to integer promotion rules: Even if the user passed a short int, short unsigned
                            // etc. - these will come in after promotion, as int's (or unsigned for the case of
                            // short unsigned when it has the same size as int)
                            int const value = (flags & EFlags::char_) != EFlags::none ? (signed char)va_arg(args, int)
                                              : (flags & EFlags::short_) != EFlags::none ? (short int)va_arg(args, int)
                                                                                         : va_arg(args, int);
                            print_integer(output, abs_for_printing(value), value < 0, base, precision, width, flags);
                        }
                    }
                    else
                    { // An unsigned specifier: u, x, X, o, b
                        flags &= ~(EFlags::plus | EFlags::space);

                        if ((flags & EFlags::long_long) != EFlags::none)
                        {
                            print_integer(output, (uint32_t)va_arg(args, unsigned long long), false, base, precision, width, flags);
                        }
                        else if (any(flags & EFlags::long_))
                        {
                            print_integer(output, (uint32_t)va_arg(args, unsigned long), false, base, precision, width, flags);
                        }
                        else
                        {
                            unsigned int const value = (flags & EFlags::char_) != EFlags::none
                                                           ? (unsigned char)va_arg(args, unsigned int)
                                                       : (flags & EFlags::short_) != EFlags::none
                                                           ? (unsigned short int)va_arg(args, unsigned int)
                                                           : va_arg(args, unsigned int);
                            print_integer(output, (uint32_t)value, false, base, precision, width, flags);
                        }
                    }
                    break;
                }
                case 'f':
                case 'F':
                    if (*format == 'F')
                        flags |= EFlags::uppercase;
                    print_floating_point(output, va_arg(args, double), precision, width, flags, prefer_decimal);
                    format++;
                    break;
                case 'e':
                case 'E':
                case 'g':
                case 'G':
                    if ((*format == 'g') || (*format == 'G'))
                        flags |= EFlags::adapt_exp;
                    if ((*format == 'E') || (*format == 'G'))
                        flags |= EFlags::uppercase;
                    print_floating_point(output, va_arg(args, double), precision, width, flags, prefer_exponential);
                    format++;
                    break;
                case 'c':
                {
                    uint32_t l = 1U;
                    // pre padding
                    if ((flags & EFlags::left) == EFlags::none)
                    {
                        while (l++ < width)
                        {
                            putchar_via_gadget(output, ' ');
                        }
                    }
                    // char output
                    putchar_via_gadget(output, (char)va_arg(args, int));
                    // post padding
                    if ((flags & EFlags::left) != EFlags::none)
                    {
                        while (l++ < width)
                        {
                            putchar_via_gadget(output, ' ');
                        }
                    }
                    format++;
                    break;
                }

                case 's':
                {
                    char const* p = va_arg(args, char*);
                    if (p == nullptr)
                    {
                        char const* f = ")llun(";
                        out_rev_(output, f, 6, width, flags);
                    }
                    else
                    {
                        uint32_t l = strnlen_s_(p, precision ? precision : max_possible_buffer_size);
                        // pre padding
                        if (any(flags & EFlags::precision))
                        {
                            l = (l < precision ? l : precision);
                        }
                        if (!any(flags & EFlags::left))
                        {
                            while (l++ < width)
                            {
                                putchar_via_gadget(output, ' ');
                            }
                        }
                        // string output
                        while ((*p != 0) && (!any(flags & EFlags::precision) || precision))
                        {
                            putchar_via_gadget(output, *(p++));
                            --precision;
                        }
                        // post padding
                        if (any(flags & EFlags::left))
                        {
                            while (l++ < width)
                            {
                                putchar_via_gadget(output, ' ');
                            }
                        }
                    }
                    format++;
                    break;
                }

                case 'p':
                {
                    width = sizeof(void*) * 2U + 2; // 2 hex chars per byte + the "0x" prefix
                    flags |= EFlags::zeropad | EFlags::pointer;
                    uintptr_t value = (uintptr_t)va_arg(args, void*);
                    (value == (uintptr_t) nullptr)
                        ? out_rev_(output, ")lin(", 5, width, flags)
                        : print_integer(output, static_cast<uint32_t>(value), false, EBase::hex, precision, width, flags);
                    format++;
                    break;
                }

                case '%':
                    putchar_via_gadget(output, '%');
                    format++;
                    break;

                case 'n':
                {
                    if (any(flags & EFlags::char_))
                        *(va_arg(args, char*)) = (char)output.pos;
                    else if (any(flags & EFlags::short_))
                        *(va_arg(args, short*)) = (short)output.pos;
                    else if (any(flags & EFlags::long_))
                        *(va_arg(args, long*)) = (long)output.pos;
                    else if (any(flags & EFlags::long_long))
                        *(va_arg(args, long long*)) = (long long int)output.pos;
                    else
                        *(va_arg(args, int*)) = (int)output.pos;
                    format++;
                    break;
                }

                default:
                    putchar_via_gadget(output, *format);
                    format++;
                    break;
            }
        }

        // termination
        append_termination_with_gadget(output);

        // return written chars without terminating \0
        return (int)output.pos;
    }
} // namespace dmt::detail

namespace dmt {
    __device__ int snprintf(char* s, size_t count, char const* format, ...)
    {
        va_list args;
        va_start(args, format);
        int const ret = vsnprintf(s, count, format, args);
        va_end(args);
        return ret;
    }

    __device__ int vsnprintf(char* s, size_t count, char const* format, va_list arg)
    {
        detail::output_gadget_t gadget{s, static_cast<int32_t>(count)};
        return detail::vsnprintf(gadget, format, arg);
    }
} // namespace dmt
