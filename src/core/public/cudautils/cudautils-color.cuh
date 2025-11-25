#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_COLOR_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_COLOR_CUH

#include "cudautils/cudautils-macro.cuh"
#include "cudautils/cudautils-vecmath.cuh"

// windows bull*
#undef RGB

namespace dmt {
    struct DMT_CORE_API RGB
    {
        float r, g, b;

        static inline RGB fromVec(Tuple3f v) { return {v.x, v.y, v.z}; }
        static inline RGB fromScalar(float f) { return {f, f, f}; }
        static inline RGB one() { return {1, 1, 1}; }

        inline float&       operator[](int32_t i) { return reinterpret_cast<float*>(this)[i]; }
        inline float const& operator[](int32_t i) const { return reinterpret_cast<float const*>(this)[i]; }

        inline Vector3f mul(RGB other) const { return {r * other.r, g * other.g, b * other.b}; }

        inline RGB saturate() const { return {fl::clamp01(r), fl::clamp01(g), fl::clamp01(b)}; }

        inline RGB saturate0() const { return {fmaxf(r, 0.f), fmaxf(g, 0.f), fmaxf(b, 0.f)}; }

        inline RGB abs() const { return {fl::abs(r), fl::abs(g), fl::abs(b)}; }

        inline float avg() const { return (r + g + b) / 3.f; }

        inline float max() const { return fmaxf(fmaxf(r, g), b); }

        inline Vector3f& asVec() { return *reinterpret_cast<Vector3f*>(this); }

        inline Vector3f const& asVec() const { return *reinterpret_cast<Vector3f const*>(this); }

        inline RGB& operator+=(const RGB& other)
        {
            r += other.r;
            g += other.g;
            b += other.b;
            return *this;
        }

        inline RGB& operator-=(const RGB& other)
        {
            r -= other.r;
            g -= other.g;
            b -= other.b;
            return *this;
        }

        inline RGB& operator*=(float scalar)
        {
            r *= scalar;
            g *= scalar;
            b *= scalar;
            return *this;
        }

        inline RGB& operator/=(float scalar)
        {
            r /= scalar;
            g /= scalar;
            b /= scalar;
            return *this;
        }

        inline RGB& operator*=(RGB c)
        {
            r *= c.r;
            g *= c.g;
            b *= c.b;
            return *this;
        }

        inline RGB& operator/=(RGB c)
        {
            r /= c.r;
            g /= c.g;
            b /= c.b;
            return *this;
        }
    };

    inline RGB  operator+(RGB first, RGB other) { return {first.r + other.r, first.g + other.g, first.b + other.b}; }
    inline RGB  operator-(RGB first, RGB other) { return {first.r - other.r, first.g - other.g, first.b - other.b}; }
    inline RGB  operator*(float scalar, RGB rgb) { return {rgb.r * scalar, rgb.g * scalar, rgb.b * scalar}; }
    inline RGB  operator/(float scalar, RGB rgb) { return {scalar / rgb.r, scalar / rgb.g, scalar / rgb.b}; }
    inline RGB  operator*(RGB rgb, float scalar) { return {rgb.r * scalar, rgb.g * scalar, rgb.b * scalar}; }
    inline RGB  operator/(RGB rgb, float scalar) { return {rgb.r / scalar, rgb.g / scalar, rgb.b / scalar}; }
    inline RGB  operator*(RGB a, RGB b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
    inline RGB  operator/(RGB a, RGB b) { return {a.r / b.r, a.g / b.g, a.b / b.b}; }
    inline RGB  lerp(RGB a, RGB b, float t) { return t <= 0.f ? a : (t >= 1.f ? b : ((1.f - t) * a + t * b)); }
    inline bool operator==(RGB rgb, RGB other) { return rgb.r == other.r && rgb.g == other.g && rgb.b == other.b; }


} // namespace dmt
#endif // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_COLOR_CUH
