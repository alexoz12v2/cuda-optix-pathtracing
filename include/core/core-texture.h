#pragma once

#include "core/core-macros.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-math.h"

namespace dmt {
    struct ImageTexture
    {
        RGB*     ptr;
        uint32_t xRes, yRes;
    };

    struct ApproxDifferentialsContext
    {
        Point3f   p;                   /// intersection point
        Vector3f  n;                   /// normal from intersection point
        Transform cameraFromRender;    /// transform from render space to camera space
        Vector3f  minPosDifferentialX; /// minimum offset in position for a ray differential along the X axis
        Vector3f  minPosDifferentialY; /// minimum offset in position for a ray differential along the Y axis
        Vector3f  minDirDifferentialX; /// minimum offset in direction for a ray differential along the X axis
        Vector3f  minDirDifferentialY; /// minimum offset in direction for a ray differential along the Y axis
        uint32_t samplesPerPixel; /// used to compute a clamped scale factor for differentials max(rsqrt(samplesPerPixel), 1/8)
    };

    /// orient camera such that z axis is aligned with the ray direction (camera position -> intersection point)
    /// camera defines min differentials in x and y, for both the origin of the ray and direction of the ray (4 vectors in total).
    /// they can be used to estimate 2 new rays and intersection points
    /// All differentials here, in `x` and `y`, are expressed in *Raster Space* (origin top left, y downwards)
    /// @note **In the current implementation this is called at every intersection**
    DMT_CORE_API void approximate_dp_dxy(ApproxDifferentialsContext const& apctx, Vector3f* dpdx, Vector3f* dpdy);

    /// All differentials here, in `x` and `y`, are expressed in *Raster Space* (origin top left, y downwards)
    struct UVDifferentialsContext
    {
        Vector3f dpdx;
        Vector3f dpdy;
        Vector3f dpdu;
        Vector3f dpdv;
    };

    /// differentials (ie speed of change) of UV coordinates with respect to change into raster space coordinates
    /// Esample of usage: compute MIP level:
    /// ```
    /// float lenX = sqrt(dudx * dudx + dvdx * dvdx); // derivative in screen-x
    /// float lenY = sqrt(dudy * dudy + dvdy * dvdy); // derivative in screen-y
    /// float footprintExtent = max(lenX, lenY);
    /// float lambda = log2(footprintExtent); // assuming normalized uv coordinates
    /// ```
    /// After selecting the MIP level, you need to sample a value from it. A good filtering procedure is Elliptic Weighted Average (EWA)
    /// - 4 values du,dv define a parallelogram into texture space representing the projection of the current pixel into texture space
    /// - copute ellipse matrix M: The implicit equation $E(u,v) = Au^2 + Buv + Cv^2$ tells you whether uv point is inside or outside ellipse (E(u,v) < 1 for inside)
    ///   ```
    ///   M = [ A   B/2 ]
    ///       [ B/2 C   ]
    ///   ```
    ///   To compute coefficients:
    ///   ```
    ///   float A = dudy*dudy + dvdy*dvdy;        // "only y"
    ///   float B = -2 * (dudx*dudy + dvdx*dvdy); // "cross"
    ///   float C = dudx*dudx + dvdx*dvdx;        // "only x"
    ///   // ellispe normalization such that largest eigenvalue is 1 (==footprint of ellipse should integrate to 1 pixel only)
    ///   float F = A * C - 0.25f * B * B;
    ///   float scale = 1.f / F;
    ///   A /= scale, B /= scale, C /= scale;
    ///   ```
    /// Once you have those, loop over all texels inside the EWA bounding box and compute a weighted average on those, where the weights are equal
    /// `weight = exp(-k * E(u,v))` where `k = 2.0` falloff coefficient
    struct UVDifferentials
    {
        float dudx;
        float dudy;
        float dvdx;
        float dvdy;
    };

    /// Apply Chain Rule: dpdx = dpdu dudx + dpdv dvdx
    /// dudx,dudy,dvdx,dvdy should be handled by BDSF evaluation and surface intersection
    DMT_CORE_API UVDifferentials duv_From_dp_dxy(UVDifferentialsContext const& uvctx);

    struct TextureEvalContext
    {
        Point3f         p;
        Vector3f        dpdx;
        Vector3f        dpdy;
        Vector3f        n; /// unit
        Point2f         uv;
        UVDifferentials dUV;
    };

    // NOTE: this texture framework, based on a tag + union and switch on texture types is the simplest which can be easily transposed onto device
    // code when the times comes
    template <typename T>
    concept Texture = requires(T const& tex, TextureEvalContext const& texEvalCtx) {
        { tex.evalFloat(texEvalCtx) } -> std::same_as<float>;
        { tex.evalRGB(texEvalCtx) } -> std::same_as<RGB>;
    };

    enum class TextureType
    {
        eImage = 0,
        eChecker,

        eCount
    };

    // -- Checker Texture --

    struct DMT_CORE_API CheckerTexture
    {
        float evalFloat(TextureEvalContext const& ctx) const;
        RGB   evalRGB(TextureEvalContext const& ctx) const;

        float scaleU;
        float scaleV;
        RGB   color1;
        RGB   color2;
    };
    static_assert(Texture<CheckerTexture>);

    enum class TexWrapMode : uint8_t
    {
        eClamp = 0,
        eRepeat,
        eMirror,
        eBlack,

        eCount
    };

    // -- Image Texture --
    inline constexpr int32_t EWA_LUT_SIZE = 128;
    extern DMT_CORE_API std::array<float, EWA_LUT_SIZE> EwaWeightLUT;

    struct DMT_CORE_API ImageTexturev2
    {
        float evalFloat(TextureEvalContext const& ctx) const;
        RGB   evalRGB(TextureEvalContext const& ctx) const;

        union Data
        {
            float*          L;
            OctahedralNorm* ns;
            RGB*            rgb;
        } data; /// mipmapped image data, stored in morton order

        int32_t     width;          /// original image x resolution (POT)
        int32_t     height;         /// original image y resolution (POT)
        int32_t     mipLevels : 30; /// number of mip levels
        int32_t     isRGB     : 1;  /// is the buffer RGB or grayscale
        int32_t     isNormal  : 1;  /// is the buffer RGB or grayscale
        int32_t     deviceIdx;      /// if -1, then CPU resident texture, otherwise index of device (TODO later)
        TexWrapMode wrapModeX;
        TexWrapMode wrapModeY;
    };

    // ---- Image Texture Utils ----
    inline constexpr float MaxAnisotropy = 8.f;

    /// Compute total number of pixels in the mip chain
    DMT_CORE_API size_t mipChainPixelCount(int w, int h);

    /// Compute offset into the mipmapped array for the start of a given level
    DMT_CORE_API size_t mortonLevelOffset(int baseW, int baseH, int level);

    /// assumes color has been already linearized and that image is stored in row major order
    DMT_CORE_API ImageTexturev2 makeRGBMipmappedTexture(
        RGB const*                 image,
        int32_t                    xRes,
        int32_t                    yRes,
        TexWrapMode                wrapModeX,
        TexWrapMode                wrapModeY,
        std::pmr::memory_resource* memory);

    /// the texture object must be initialized and the memory reosurce given must be the same with which you created it
    DMT_CORE_API void freeImageTexture(ImageTexturev2& tex, std::pmr::memory_resource* memory);

    // -- Texture Dispatch Type --
    union DMT_CORE_API TexturePayload
    {
        ImageTexturev2 image;
        CheckerTexture checker;
    };

    struct DMT_CORE_API TextureVariant
    {
        TextureType    type;
        TexturePayload payload;

        float evalFloat(TextureEvalContext const& ctx) const;
        RGB   evalRGB(TextureEvalContext const& ctx) const;
    };
} // namespace dmt