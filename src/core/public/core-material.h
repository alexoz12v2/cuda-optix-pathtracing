#ifndef DMT_CORE_PUBLIC_CORE_MATERIAL_H
#define DMT_CORE_PUBLIC_CORE_MATERIAL_H

#include "core-macros.h"
#include "core-bsdf.h"
#include "core-texture.h"
#include "core-texture-cache.h"

namespace dmt {
    struct alignas(16) SurfaceMaterial
    {
        /// When shading a dielectric, this accounts for `weight` in GGX, `albedo` in Oren Nayar (oren nayar uses 1,1,1 as weight)
        /// When shading a conductor, this accounts for `weight` (less important than eta and etak, which are fixed (TODO: see correctness))
        ///   Only if metallic is 1.0f
        /// hashCRC64 of diffuse texture path or 0
        uint64_t diffusekey;
        uint32_t diffuseWidth;
        uint32_t diffuseHeight;

        /// Shading normals
        /// hashCRC64 of normal texture path or 0
        uint64_t normalkey;
        uint32_t normalWidth;
        uint32_t normalHeight;

        /// weight for linear interpolation between a conductor contribution and dielectric contribution
        /// when this is different than 0 or 1, diffuse will be given to the dielectric counterpart, and {1,1,1} will be set
        /// for conductor
        /// hashCRC64 of metallic texture path or 0
        uint64_t metallickey;
        uint32_t metallicWidth;
        uint32_t metallicHeight;

        /// roughness, remapped, through a sqrt function or similiar, to either alphax and alphay of anisotropic GGX together
        /// with anisotropy parameter, or roughness of Oren Nayar
        /// hashCRC64 of roughness texture path or zero
        uint64_t roughnesskey;
        uint32_t roughnessWidth;
        uint32_t roughnessHeight;

        /// xx'yy'zz'ww
        /// if x = 1 -> diffuse uses texture, y = 1 -> normal uses texture, z = 1 -> metallic uses texture, w = 1 -> roughness uses texture
        int32_t                  texMatMap;
        static constexpr int32_t MetallicMask  = 0x00'00'ff'00;
        static constexpr int32_t DiffuseMask   = 0xff'00'00'00;
        static constexpr int32_t NormalMask    = 0x00'ff'00'00;
        static constexpr int32_t RoughnessMask = 0x00'00'00'ff;

        /// fallback of metallicKey when it is equal to zero
        float metallicvalue;

        /// fallback of roughnessKey when it is equal to zero
        float roughnessvalue;

        /// fallback of diffuseKey when it is equal to zero (Byte encoded, RGB B -> LSB)
        uint32_t diffusevalue;

        /// multiplier for roughness in the x (tangent) direction over y (bitangent) (GGX only) (max 8)
        float anisotropy;

        /// multiplier for amplifying multiple scatter contributions (Oren Nayar Only)
        float multiscatterMultiplier;

        /// distriminator to know whether to use Oren Nayar of GGX for dielectric part of the material
        uint32_t isDiffuseOpaque   : 31;
        uint32_t useShadingNormals : 1;

        /// relative (to air) ior for dielectric (GGX)
        float ior;

        /// relative (to air) ior for conductor (real part)
        RGB eta;

        /// relative (to air) ior for conductor (complex part)
        RGB etak;

        /// reflectance tint parameter for GGX dielectric
        RGB reflectanceTint;

        /// transmittance tint parameter for GGX dielectric
        RGB transmittanceTint;

        /// fallback of normalKey when it is equal to zero
        OctahedralNorm normalvalue;
    };
    static_assert(alignof(SurfaceMaterial) == 16 && std::is_standard_layout_v<SurfaceMaterial>);

    DMT_CORE_API BSDFSample materialSample(
        SurfaceMaterial const&    material,
        TextureCache&             cache,
        TextureEvalContext const& texCtx, /// n to represent shading normal
        Vector3f                  w,
        Vector3f                  ng,
        Point2f                   u,
        float                     uc);

    DMT_CORE_API BSDFEval materialEval(SurfaceMaterial const&    material,
                                       TextureCache&             cache,
                                       TextureEvalContext const& texCtx,
                                       Vector3f                  wo,
                                       Vector3f                  wi,
                                       Vector3f                  ng);

    DMT_CORE_API Normal3f materialShadingNormal(SurfaceMaterial const&    material,
                                                TextureCache&             cache,
                                                TextureEvalContext const& texCtx,
                                                Vector3f                  ng);

    inline RGB rgbFromByte3(uint32_t encoded)
    {
        RGB res{};
        res.r = static_cast<float>((encoded >> 16) & 0xFF) / 255.f;
        res.g = static_cast<float>((encoded >> 8) & 0xFF) / 255.f;
        res.b = static_cast<float>(encoded & 0xFF) / 255.f;
        return res;
    }

    inline uint32_t byte3FromRGB(RGB const& c)
    {
        uint32_t r = static_cast<uint32_t>(fl::clamp(c.r, 0.f, 1.f) * 255.f + 0.5f);
        uint32_t g = static_cast<uint32_t>(fl::clamp(c.g, 0.f, 1.f) * 255.f + 0.5f);
        uint32_t b = static_cast<uint32_t>(fl::clamp(c.b, 0.f, 1.f) * 255.f + 0.5f);

        return (r << 16) | (g << 8) | b;
    }
} // namespace dmt
#endif // DMT_CORE_PUBLIC_CORE_MATERIAL_H
