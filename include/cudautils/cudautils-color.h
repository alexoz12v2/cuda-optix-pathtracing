#pragma once

#include "cudautils/cudautils-macro.h"

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

        inline float avg() const { return (r + g + b) / 3.f; }

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
    inline bool operator==(RGB rgb, RGB other) { return rgb.r == other.r && rgb.g == other.g && rgb.b == other.b; }

#if 0
    // RGBColorSpace Definition
    class RGBColorSpace
    {
    public:
        // RGBColorSpace Public Methods
        RGBColorSpace(Point2f                   r,
                      Point2f                   g,
                      Point2f                   b,
                      Spectrum                  illuminant,
                      RGBToSpectrumTable const* rgbToSpectrumTable,
                      Allocator                 alloc);

        DMT_CPU_GPU RGBSigmoidPolynomial ToRGBCoeffs(RGB rgb) const;

        static void Init(Allocator alloc);

        // RGBColorSpace Public Members
        Point2f                     r, g, b, w;
        DenselySampledSpectrum      illuminant;
        SquareMatrix<3>             XYZFromRGB, RGBFromXYZ;
        static RGBColorSpace const *sRGB, *DCI_P3, *Rec2020, *ACES2065_1;

        DMT_CPU_GPU bool operator==(RGBColorSpace const& cs) const
        {
            return (r == cs.r && g == cs.g && b == cs.b && w == cs.w && rgbToSpectrumTable == cs.rgbToSpectrumTable);
        }
        DMT_CPU_GPU bool operator!=(RGBColorSpace const& cs) const
        {
            return (r != cs.r || g != cs.g || b != cs.b || w != cs.w || rgbToSpectrumTable != cs.rgbToSpectrumTable);
        }

        std::string ToString() const;

        DMT_CPU_GPU RGB LuminanceVector() const { return RGB(XYZFromRGB[1][0], XYZFromRGB[1][1], XYZFromRGB[1][2]); }

        DMT_CPU_GPU RGB ToRGB(XYZ xyz) const { return Mul<RGB>(RGBFromXYZ, xyz); }
        DMT_CPU_GPU XYZ ToXYZ(RGB rgb) const { return Mul<XYZ>(XYZFromRGB, rgb); }

        static RGBColorSpace const* GetNamed(std::string name);
        static RGBColorSpace const* Lookup(Point2f r, Point2f g, Point2f b, Point2f w);

    private:
        // RGBColorSpace Private Members
        RGBToSpectrumTable const* rgbToSpectrumTable;
    };

    #ifdef PBRT_BUILD_GPU_RENDERER
    extern PBRT_CONST RGBColorSpace* RGBColorSpace_sRGB;
    extern PBRT_CONST RGBColorSpace* RGBColorSpace_DCI_P3;
    extern PBRT_CONST RGBColorSpace* RGBColorSpace_Rec2020;
    extern PBRT_CONST RGBColorSpace* RGBColorSpace_ACES2065_1;
    #endif

    SquareMatrix<3> ConvertRGBColorSpace(RGBColorSpace const& from, RGBColorSpace const& to);
#endif

} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_COLOR_IMPL)
    #include "cudautils-color.cu"
#endif