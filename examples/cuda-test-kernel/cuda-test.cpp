#include "cuda-test.h"

#include "platform/platform-context.h"
#include <algorithm>

namespace dmt {
    // !! TODO: copy cuda/std in build directory!
    static std::pair<std::vector<char const*>, std::vector<char const*>> getNvrtcHeaderOverrides()
    {
        // clang-format off
        return std::make_pair<std::vector<char const*>, std::vector<char const*>>({
            "cassert",
            "cstddef",
            "cmath",
            "cstdint",
            "stdint.h",
            "type_traits",
            "climits",
            "cfloat",
            "limits",
            "stdlib.h",
            "complex",
            "cerrno",
            "cstdlib",
            "functional",
            "sstream",
            "iosfwd",
            "cstring",
            "string.h",
            "string",
            "algorithm",
            "array",
            "numbers"
        }, {
            R"(
#pragma once
#ifndef assert
    #if !defined(DMT_DEBUG)
        #define assert(cond) if (!(cond)) asm(\"trap;\")
    #endif
#endif
            )",
            R"(
#pragma once
typedef decltype(sizeof(0)) size_t;
typedef decltype(nullptr) nullptr_t;
typedef unsigned char byte;
namespace std {
    typedef decltype(sizeof(0)) size_t;
    typedef decltype(nullptr) nullptr_t;
    typedef unsigned char byte;
}
            )",
            R"(
#pragma once
namespace std {
    __device__ inline float sqrt(float x) { return ::sqrtf(x); }
    __device__ inline float sin(float x)  { return ::sinf(x); }
    __device__ inline float cos(float x)  { return ::cosf(x); }
    __device__ inline float tan(float x)  { return ::tanf(x); }
    __device__ inline float asin(float x) { return ::asinf(x); }
    __device__ inline float asinh(float x) { return ::asinhf(x); }
    __device__ inline float acos(float x) { return ::acosf(x); }
    __device__ inline float acosh(float x) { return ::acoshf(x); }
    __device__ inline float atan(float x) { return ::atanf(x); }
    __device__ inline float atanh(float x) { return ::atanhf(x); }
    __device__ inline float sinh(float x) { return ::sinhf(x); }
    __device__ inline float cosh(float x) { return ::coshf(x); }
    __device__ inline float tanh(float x) { return ::tanhf(x); }
    __device__ inline float exp(float x)  { return ::expf(x); }
    __device__ inline float exp2(float x) { return ::exp2f(x); }
    __device__ inline float log(float x)  { return ::logf(x); }
    __device__ inline float log2(float x) { return ::log2f(x); }
    __device__ inline float pow(float x, float y) { return ::powf(x, y); }
    __device__ inline float round(float x) { return ::roundf(x); }
    __device__ inline float trunc(float x) { return ::truncf(x); }
    __device__ inline float floor(float x) { return ::floorf(x); }
    __device__ inline float ceil(float x)  { return ::ceilf(x); }
    __device__ inline float fma(float a, float b, float c) { return ::fmaf(a,b,c); }
    __device__ inline bool isnan(float x) { return ::isnan(x); }
    __device__ inline bool isinf(float x) { return ::isinf(x); }
} // namespace std
            )",
            "#pragma once\n"
            "typedef unsigned long long uintptr_t;\n"
            "typedef signed char        int8_t;\n"
            "typedef short              int16_t;\n"
            "typedef int                int32_t;\n"
            "typedef long long          int64_t;\n"
            "typedef unsigned char      uint8_t;\n"
            "typedef unsigned short     uint16_t;\n"
            "typedef unsigned int       uint32_t;\n"
            "typedef unsigned long long uint64_t;\n"
            "namespace std { using int8_t = ::int8_t; using int16_t = ::int16_t; using int32_t = ::int32_t;\n"
            "  using int64_t = ::int64_t; using uint8_t = ::uint8_t; using uint16_t = ::uint16_t;\n"
            "  using uint32_t = ::uint32_t; using uint64_t = ::uint64_t;\n"
            "}\n",
            "#pragma once\n#include <cstdint>",
            R"(
#pragma once

namespace std {

// remove const
template<typename T> struct remove_const      { using type = T; };
template<typename T> struct remove_const<const T> { using type = T; };

// remove reference
template<typename T> struct remove_reference   { using type = T; };
template<typename T> struct remove_reference<T&> { using type = T; };

// is_same
template<typename A, typename B> struct is_same { static const bool value = false; };
template<typename A> struct is_same<A,A> { static const bool value = true; };

// conditional
template<bool B, typename T, typename F> struct conditional { using type = T; };
template<typename T, typename F> struct conditional<false,T,F> { using type = F; };

// enable_if
template<bool B, typename T = void> struct enable_if {};
template<typename T> struct enable_if<true,T> { using type = T; };

// is_integral
template<typename T> struct is_integral { static const bool value = false; };
template<> struct is_integral<int>   { static const bool value = true; };
template<> struct is_integral<unsigned int> { static const bool value = true; };
template<> struct is_integral<long>  { static const bool value = true; };
template<> struct is_integral<unsigned long> { static const bool value = true; };
template<> struct is_integral<short> { static const bool value = true; };
template<> struct is_integral<unsigned short> { static const bool value = true; };
template<> struct is_integral<char>  { static const bool value = true; };
template<> struct is_integral<unsigned char> { static const bool value = true; };

// is_floating_point
template<typename T> struct is_floating_point { static const bool value = false; };
template<> struct is_floating_point<float>  { static const bool value = true; };
template<> struct is_floating_point<double> { static const bool value = true; };

// make_unsigned
template<typename T> struct make_unsigned {};
template<> struct make_unsigned<char>   { using type = unsigned char; };
template<> struct make_unsigned<short>  { using type = unsigned short; };
template<> struct make_unsigned<int>    { using type = unsigned int; };
template<> struct make_unsigned<long>   { using type = unsigned long; };
template<> struct make_unsigned<long long> { using type = unsigned long long; };

} // namespace std
            )",
            R"(
#pragma once

#define CHAR_BIT   8

#define SCHAR_MIN  (-128)
#define SCHAR_MAX  127
#define UCHAR_MAX  255

#define CHAR_MIN   SCHAR_MIN
#define CHAR_MAX   SCHAR_MAX

#define SHRT_MIN   (-32768)
#define SHRT_MAX   32767
#define USHRT_MAX  65535

#define INT_MIN    (-2147483647-1)
#define INT_MAX    2147483647
#define UINT_MAX   4294967295U

#define LONG_MIN   (-9223372036854775807L-1)
#define LONG_MAX   9223372036854775807L
#define ULONG_MAX  18446744073709551615UL

#define LLONG_MIN  (-9223372036854775807LL-1)
#define LLONG_MAX  9223372036854775807LL
#define ULLONG_MAX 18446744073709551615ULL
            )",
            R"(
#pragma once

// https://stackoverflow.com/questions/8812422/how-to-find-epsilon-min-and-max-constants-for-cuda
// Single-precision (float) limits
#define FLT_RADIX       2
#define FLT_MANT_DIG    24
#define FLT_DIG         6
#define FLT_MIN_EXP     (-125)
#define FLT_MIN_10_EXP  (-37)
#define FLT_MAX_EXP     128
#define FLT_MAX_10_EXP  38
#define FLT_MAX         3.402823466e+38F
#define FLT_MIN         1.175494351e-38F
#define FLT_EPSILON     1.192092896e-07F
#define FLT_DENORM_MIN  1.401298464e-45F

// Double-precision (double) limits
#define DBL_MANT_DIG    53
#define DBL_DIG         15
#define DBL_MIN_EXP     (-1021)
#define DBL_MIN_10_EXP  (-307)
#define DBL_MAX_EXP     1024
#define DBL_MAX_10_EXP  308
#define DBL_MAX         1.7976931348623157e+308
#define DBL_MIN         2.2250738585072014e-308
#define DBL_EPSILON     2.2204460492503131e-16
#define DBL_DENORM_MIN  4.9406564584124654e-324

// Long double (usually same as double on NVRTC)
#define LDBL_MANT_DIG   DBL_MANT_DIG
#define LDBL_DIG        DBL_DIG
#define LDBL_MIN_EXP    DBL_MIN_EXP
#define LDBL_MIN_10_EXP DBL_MIN_10_EXP
#define LDBL_MAX_EXP    DBL_MAX_EXP
#define LDBL_MAX_10_EXP DBL_MAX_10_EXP
#define LDBL_MAX        DBL_MAX
#define LDBL_MIN        DBL_MIN
#define LDBL_EPSILON    DBL_EPSILON
#define LDBL_DENORM_MIN DBL_DENORM_MIN 
            )",
            R"(
#pragma once

#include "climits"   // depends on our climits stub
#include "cfloat"    // depends on our cfloat stub

#ifndef __builtin_inff
__host__ __device__ inline float __builtin_inff()
{
    return __int_as_float(0x7f800000); // IEEE-754 +inf
}
#endif

#ifndef __builtin_inf
__host__ __device__ inline double __builtin_inf()
{
    return __longlong_as_double(0x7ff0000000000000ULL); // IEEE-754 +inf
}
#endif

namespace std {

template<typename T> struct numeric_limits {
    static constexpr bool is_specialized = false;
};

__host__ __device__ inline float __nanf() {
    return __int_as_float(0x7fffffff); // quiet NaN
}

__host__ __device__ inline double __nand() {
    return __longlong_as_double(0x7fffffffffffffffULL); // quiet NaN
}

// float
template<> struct numeric_limits<float> {
    static constexpr bool is_specialized = true;
    static constexpr bool is_iec559 = true;
    static constexpr float min() noexcept { return FLT_MIN; }
    static constexpr float max() noexcept { return FLT_MAX; }
    static constexpr float lowest() noexcept { return -FLT_MAX; }
    static constexpr int digits = FLT_MANT_DIG;
    static constexpr int digits10 = FLT_DIG;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr float epsilon() noexcept { return FLT_EPSILON; }
    static constexpr float infinity() noexcept { return __builtin_inff(); }
    __host__ __device__ static float quiet_NaN() noexcept { return __nanf(); }
    __host__ __device__ static float signaling_NaN() noexcept { return __nanf(); } // usually same bit pattern
};

// double
template<> struct numeric_limits<double> {
    static constexpr bool is_specialized = true;
    static constexpr bool is_iec559 = true;
    static constexpr double min() noexcept { return DBL_MIN; }
    static constexpr double max() noexcept { return DBL_MAX; }
    static constexpr double lowest() noexcept { return -DBL_MAX; }
    static constexpr int digits = DBL_MANT_DIG;
    static constexpr int digits10 = DBL_DIG;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr double epsilon() noexcept { return DBL_EPSILON; }
    static constexpr double infinity() noexcept { return __builtin_inf(); }
    __host__ __device__ static double quiet_NaN() noexcept { return __nand(); }
    __host__ __device__ static double signaling_NaN() noexcept { return __nand(); }
};

// int
template<> struct numeric_limits<int> {
    static constexpr bool is_specialized = true;
    static constexpr int min() noexcept { return INT_MIN; }
    static constexpr int max() noexcept { return INT_MAX; }
    static constexpr int lowest() noexcept { return INT_MIN; }
    static constexpr int digits = 31;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = true;
};

// unsigned int
template<> struct numeric_limits<unsigned int> {
    static constexpr bool is_specialized = true;
    static constexpr unsigned int min() noexcept { return 0; }
    static constexpr unsigned int max() noexcept { return UINT_MAX; }
    static constexpr unsigned int lowest() noexcept { return 0; }
    static constexpr int digits = 32;
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = true;
};

// Add other types as needed: short, long, long long, char, etc.

} // namespace std
            )",
            R"(
            )",
            R"(
#pragma once

namespace std {

// Minimal std::complex for float and double
template<typename T>
struct complex {
    T real_;
    T imag_;

    __device__ constexpr complex(T r = T{}, T i = T{}) : real_(r), imag_(i) {}
    
    __device__ T real() const { return real_; }
    __device__ T imag() const { return imag_; }

    __device__ void real(T r) { real_ = r; }
    __device__ void imag(T i) { imag_ = i; }

    // basic arithmetic
    __device__ complex<T>& operator+=(const complex<T>& o) {
        real_ += o.real_;
        imag_ += o.imag_;
        return *this;
    }

    __device__ complex<T>& operator-=(const complex<T>& o) {
        real_ -= o.real_;
        imag_ -= o.imag_;
        return *this;
    }

    __device__ complex<T>& operator*=(const complex<T>& o) {
        T r = real_ * o.real_ - imag_ * o.imag_;
        T i = real_ * o.imag_ + imag_ * o.real_;
        real_ = r; imag_ = i;
        return *this;
    }

    __device__ complex<T>& operator/=(const complex<T>& o) {
        T denom = o.real_ * o.real_ + o.imag_ * o.imag_;
        T r = (real_ * o.real_ + imag_ * o.imag_) / denom;
        T i = (imag_ * o.real_ - real_ * o.imag_) / denom;
        real_ = r; imag_ = i;
        return *this;
    }
};

// free functions
template<typename T>
__device__ inline complex<T> operator+(complex<T> a, const complex<T>& b) { return a += b; }

template<typename T>
__device__ inline complex<T> operator-(complex<T> a, const complex<T>& b) { return a -= b; }

template<typename T>
__device__ inline complex<T> operator*(complex<T> a, const complex<T>& b) { return a *= b; }

template<typename T>
__device__ inline complex<T> operator/(complex<T> a, const complex<T>& b) { return a /= b; }

} // namespace std
            )",
            "#pragma once\n#define errno 0",
            "#pragma once\n#include <stdlib.h>",
            R"(
#pragma once

namespace std {

// minimal hash for int, float, double
template<typename T>
struct hash;

template<>
struct hash<int> {
    __device__ size_t operator()(int x) const noexcept { return static_cast<size_t>(x); }
};

template<>
struct hash<unsigned int> {
    __device__ size_t operator()(unsigned int x) const noexcept { return static_cast<size_t>(x); }
};

template<>
struct hash<float> {
    __device__ size_t operator()(float x) const noexcept {
        unsigned int u;
        asm("mov.b32 %0, %1;" : "=r"(u) : "f"(x));
        return static_cast<size_t>(u);
    }
};

template<>
struct hash<double> {
    __device__ size_t operator()(double x) const noexcept {
        unsigned long long u;
        asm("mov.b64 %0, %1;" : "=l"(u) : "d"(x));
        return static_cast<size_t>(u);
    }
};

// minimal function objects: plus, minus, multiplies
template<typename T>
struct plus {
    __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct minus {
    __device__ T operator()(const T& a, const T& b) const { return a - b; }
};

template<typename T>
struct multiplies {
    __device__ T operator()(const T& a, const T& b) const { return a * b; }
};

template<typename T>
struct divides {
    __device__ T operator()(const T& a, const T& b) const { return a / b; }
};

} // namespace std
            )",
            "",
            "",
            R"(
#pragma once

namespace std {

// memcpy
__device__ inline void* memcpy(void* dst, const void* src, size_t n) {
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    for (size_t i = 0; i < n; ++i) d[i] = s[i];
    return dst;
}

// memmove
__device__ inline void* memmove(void* dst, const void* src, size_t n) {
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    if (d < s) {
        for (size_t i = 0; i < n; ++i) d[i] = s[i];
    } else {
        for (size_t i = n; i != 0; --i) d[i-1] = s[i-1];
    }
    return dst;
}

// memset
__device__ inline void* memset(void* s, int c, size_t n) {
    unsigned char* p = static_cast<unsigned char*>(s);
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<unsigned char>(c);
    return s;
}

// memcmp
__device__ inline int memcmp(const void* s1, const void* s2, size_t n) {
    const unsigned char* p1 = static_cast<const unsigned char*>(s1);
    const unsigned char* p2 = static_cast<const unsigned char*>(s2);
    for (size_t i = 0; i < n; ++i) {
        if (p1[i] != p2[i]) return (p1[i] < p2[i]) ? -1 : 1;
    }
    return 0;
}

} // namespace std
            )",
            "#pragma once\n#include <cstring>",
            R"(
#pragma once

namespace std {

// Dummy string class for NVRTC device compilation
class string {
public:
    __device__ string() {}
    __device__ string(const char*) {}
    __device__ string(const string&) {}
    __device__ string& operator=(const string&) { return *this; }
    __device__ ~string() {}
    
    __device__ size_t size() const { return 0; }
    __device__ const char* c_str() const { return ""; }
};

} // namespace std
            )",
            "",
            R"(
#pragma once

namespace std {

template<typename T, size_t N>
struct array {
    T elems[N];

    __device__ constexpr T& operator[](size_t i) { return elems[i]; }
    __device__ constexpr const T& operator[](size_t i) const { return elems[i]; }

    __device__ constexpr T* data() { return elems; }
    __device__ constexpr const T* data() const { return elems; }

    __device__ constexpr size_t size() const { return N; }

    // basic front/back
    __device__ constexpr T& front() { return elems[0]; }
    __device__ constexpr const T& front() const { return elems[0]; }

    __device__ constexpr T& back() { return elems[N-1]; }
    __device__ constexpr const T& back() const { return elems[N-1]; }
};

// equality operators (optional)
template<typename T, size_t N>
__device__ constexpr bool operator==(const array<T,N>& a, const array<T,N>& b) {
    for (size_t i = 0; i < N; ++i) {
        if (a.elems[i] != b.elems[i]) return false;
    }
    return true;
}

template<typename T, size_t N>
__device__ constexpr bool operator!=(const array<T,N>& a, const array<T,N>& b) {
    return !(a == b);
}

} // namespace std 
            )",
            R"(
#pragma once

namespace std {
namespace numbers {

// Primary template for constants
template <typename T>
inline constexpr T e_v = T(2.71828182845904523536L);

template <typename T>
inline constexpr T log2e_v = T(1.44269504088896340736L);

template <typename T>
inline constexpr T log10e_v = T(0.43429448190325182765L);

template <typename T>
inline constexpr T pi_v = T(3.14159265358979323846L);

template <typename T>
inline constexpr T inv_pi_v = T(0.31830988618379067154L);

template <typename T>
inline constexpr T inv_sqrtpi_v = T(0.56418958354775628695L);

template <typename T>
inline constexpr T ln2_v = T(0.69314718055994530942L);

template <typename T>
inline constexpr T ln10_v = T(2.30258509299404568402L);

template <typename T>
inline constexpr T sqrt2_v = T(1.41421356237309504880L);

template <typename T>
inline constexpr T sqrt3_v = T(1.73205080756887729353L);

template <typename T>
inline constexpr T inv_sqrt3_v = T(0.57735026918962576451L);

template <typename T>
inline constexpr T egamma_v = T(0.57721566490153286060L);

template <typename T>
inline constexpr T phi_v = T(1.61803398874989484820L);

// Default double-precision constants (C++20 style)
inline constexpr double e        = e_v<double>;
inline constexpr double log2e    = log2e_v<double>;
inline constexpr double log10e   = log10e_v<double>;
inline constexpr double pi       = pi_v<double>;
inline constexpr double inv_pi   = inv_pi_v<double>;
inline constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
inline constexpr double ln2      = ln2_v<double>;
inline constexpr double ln10     = ln10_v<double>;
inline constexpr double sqrt2    = sqrt2_v<double>;
inline constexpr double sqrt3    = sqrt3_v<double>;
inline constexpr double inv_sqrt3 = inv_sqrt3_v<double>;
inline constexpr double egamma   = egamma_v<double>;
inline constexpr double phi      = phi_v<double>;

} // namespace numbers
} // namespace std
            )"
        });
        // clang-format on
    }

    std::vector<char const*> getnvccOpts(bool debug)
    {
        std::vector<char const*> opts{
            "--gpu-architecture=compute_60", // TODO check compatibility with current context device
            "--use_fast_math",
            "--relocatable-device-code=true",
            "--std=c++17",

            //"--define-macro=__BYTE_ORDER__=__ORDER_LITTLE_ENDIAN__",
            //"--define-macro=__ORDER_LITTLE_ENDIAN__=1234",
            //"--define-macro=__ORDER_BIG_ENDIAN__=4321",
            //"--define-macro=__ORDER_PDP_ENDIAN__=3412",
            //"--include-path=" CUDA_HOME,
            //"--include-path=" CUDA_HOME "/cuda/std",
            //"--include-path=" DMT_CPP_INCLUDE_PATH,

            "-default-device" // TODO remove or better, parametrize
        };

        if (debug)
        {
            opts.push_back("-lineinfo");
            opts.push_back("-G");
        }

        return opts;
    }

    std::unique_ptr<char[]> compilePTX(dmt::os::Path const&            path,
                                       NVRTCLibrary*                   nvrtcApi,
                                       std::string_view                kernelFileName,
                                       std::vector<char const*> const& nvccOpts)
    {
        Context ctx;

        std::ifstream file{path.toUnderlying().c_str()};

        if (!file)
        {
            ctx.log("File not found", {});
            return nullptr;
        }
        std::string srcKernel{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        return compilePTX(srcKernel, nvrtcApi, kernelFileName, nvccOpts);
    }

    std::unique_ptr<char[]> compilePTX(std::string_view                srcKernel,
                                       NVRTCLibrary*                   nvrtcApi,
                                       std::string_view                kernelFileName,
                                       std::vector<char const*> const& nvccOpts)
    {
        Context ctx;

        auto const [headerNames, headerSrc] = getNvrtcHeaderOverrides();

        nvrtcProgram prog = nullptr;
        if (auto res = nvrtcApi->nvrtcCreateProgram(&prog,
                                                    srcKernel.data(),
                                                    kernelFileName.data(),
                                                    static_cast<int>(headerNames.size()),
                                                    headerSrc.data(),
                                                    headerNames.data());
            res != ::NVRTC_SUCCESS)
        {
            ctx.error("({}) nvrtcCreateProgram Error: {}",
                      std::make_tuple(kernelFileName, nvrtcApi->nvrtcGetErrorString(res)));
            return nullptr;
        }

        std::ranges::for_each(nvccOpts, [&ctx](char const* str) {
            ctx.log("  opts: {}", std::make_tuple(std::string_view(str)));
        });

        if (nvrtcApi->nvrtcCompileProgram(prog, static_cast<int32_t>(nvccOpts.size()), nvccOpts.data()) != ::NVRTC_SUCCESS)
        {
            size_t logSize = 0;
            nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            auto logBuf = std::make_unique<char[]>(logSize + 1);
            if (!logBuf)
                return nullptr;
            logBuf[logSize] = '\0';
            nvrtcApi->nvrtcGetProgramLog(prog, logBuf.get());

            std::string logStr(logBuf.get());

            // Break the log into chunks of 256 characters
            std::istringstream iss(logStr);
            std::string        line;
            ctx.error("({}) nvrtcCompileProgram Failed:\n", std::make_tuple(kernelFileName));
            while (std::getline(iss, line))
            {
                if (!line.empty())
                {
                    if (line.find("warning:") != decltype(line)::npos || line.find("warning #") != decltype(line)::npos)
                        ctx.warn("{}", std::make_tuple(line));
                    else
                        ctx.error("{}", std::make_tuple(line));
                }
            }


            return nullptr;
        }

#if 1
        // TODO Remove: Dump compiled file to current working directory such that debugger picks on it
        std::ofstream ptxFile{"saxpy.cu"};
        ptxFile << srcKernel;
        ptxFile.flush();
#endif
        size_t ptxSize = 0;

        if (auto res = nvrtcApi->nvrtcGetPTXSize(prog, &ptxSize); res != ::NVRTC_SUCCESS)
        {
            ctx.log("{}", std::make_tuple(nvrtcApi->nvrtcGetErrorString(res)));
            return nullptr;
        }

        std::unique_ptr<char[]> ptxBuffer = std::make_unique<char[]>(ptxSize);
        nvrtcApi->nvrtcGetPTX(prog, ptxBuffer.get());
        nvrtcApi->nvrtcDestroyProgram(&prog);

        return ptxBuffer;
    }

    // ------------------------------------------------------------
    // Precompute the filter distribution in host memory
    // ------------------------------------------------------------
    PiecewiseConstant2D precalculateMitchellDistrib(filtering::Mitchell const& filter,
                                                    int                        Nx,
                                                    int                        Ny,
                                                    std::pmr::memory_resource* mem)
    {
        Bounds2f domain;
        domain.pMin = Point2f(-filter.radius().x, -filter.radius().y);
        domain.pMax = Point2f(filter.radius().x, filter.radius().y);

        dstd::Array2D<float> values(Nx, Ny);

        Vector2f cellSize((domain.pMax.x - domain.pMin.x) / Nx, (domain.pMax.y - domain.pMin.y) / Ny);

        for (int y = 0; y < Ny; ++y)
        {
            for (int x = 0; x < Nx; ++x)
            {
                // Cell center
                float px = domain.pMin.x + (x + 0.5f) * cellSize.x;
                float py = domain.pMin.y + (y + 0.5f) * cellSize.y;

                float val    = filter.evaluate(Point2f(px, py));
                values(x, y) = std::max(0.0f, val);
            }
        }

        return PiecewiseConstant2D(values, domain, mem);
    }

    GpuSamplerHandle uploadFilterDistrib(CUDADriverLibrary*         cudaApi,
                                         PiecewiseConstant2D const& cpuDistrib,
                                         filtering::Mitchell const& cpuFilter)
    {
        //device pointer
        GpuSamplerHandle handle;

        int Nx = cpuDistrib.resolution().x;
        int Ny = cpuDistrib.resolution().y;

        // Flatten conditional CDF (Nx+1 per row)
        std::vector<float> conditionalFlat;
        conditionalFlat.reserve(Ny * (Nx + 1));
        for (int y = 0; y < Ny; ++y)
        {
            auto const& rowCdf = cpuDistrib.conditionalCdfRow(y);
            conditionalFlat.insert(conditionalFlat.end(), rowCdf.begin(), rowCdf.end());
        }

        // Flatten marginal CDF (Ny+1)
        auto const& marginalCdf = cpuDistrib.marginalCdf();

        // Allocate GPU memory
        cudaApi->cuMemAlloc((CUdeviceptr*) &handle.dConditionalCdf, conditionalFlat.size() * sizeof(float));
        cudaApi->cuMemAlloc((CUdeviceptr*) &handle.dMarginalCdf, marginalCdf.size() * sizeof(float));

        // Copy to GPU
        cudaApi->cuMemcpyHtoD((CUdeviceptr)handle.dConditionalCdf, conditionalFlat.data(), conditionalFlat.size() * sizeof(float));
        cudaApi->cuMemcpyHtoD((CUdeviceptr)handle.dMarginalCdf, marginalCdf.data(), marginalCdf.size() * sizeof(float));

        // Fill GPU sampler struct
        handle.sampler.filter.radiusX = cpuFilter.radius().x;
        handle.sampler.filter.radiusY = cpuFilter.radius().y;
        handle.sampler.filter.B       = cpuFilter.b();
        handle.sampler.filter.C       = cpuFilter.c();

        handle.sampler.distrib.conditionalCdf = handle.dConditionalCdf;
        handle.sampler.distrib.marginalCdf    = handle.dMarginalCdf;
        handle.sampler.distrib.Nx             = Nx;
        handle.sampler.distrib.Ny             = Ny;
        handle.sampler.distrib.pMin           = Point2f(cpuDistrib.domain().pMin.x, cpuDistrib.domain().pMin.y);
        handle.sampler.distrib.pMax           = Point2f(cpuDistrib.domain().pMax.x, cpuDistrib.domain().pMax.y);
        handle.sampler.distrib.integral       = cpuDistrib.integral();

        return handle;
    }


} // namespace dmt
