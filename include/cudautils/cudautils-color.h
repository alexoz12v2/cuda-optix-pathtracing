#pragma once

#include "dmtmacros.h"

// windows bull*
#undef RGB

namespace dmt {
    struct RGB
    {
        float r, g, b;
    };

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


} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_COLOR_IMPL)
#include "cudautils-color.cu"
#endif