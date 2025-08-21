#include "core-parser.h"

#include "platform/platform-context.h"

// json parsing
#include <nlohmann/json.hpp>

// image parsing
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <ImfRgbaFile.h>
#include <ImfArray.h>

// C++ standard
#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace dmt {

    struct TexInfo
    {
        std::string texType;
        std::string texPath;
        uint32_t    xRes;
        uint32_t    yRes;
    };

    struct ParserState
    {
        std::unordered_map<std::string, Transform> transforms;

        std::unordered_map<std::string, Light>    lights;
        std::unordered_map<std::string, TexInfo>  texturesPath;
        std::unordered_map<std::string, uint32_t> materials;
        std::unordered_map<std::string, uint32_t> objects;
    };
} // namespace dmt

namespace dmt::dict {
    using namespace std::string_view_literals;
    // Objects
    SText camera = "camera"sv;
    SText film   = "film"sv;
    SText world  = "world"sv;
    // Array
    SText textures   = "textures"sv;
    SText materials  = "materials"sv;
    SText objects    = "objects"sv;
    SText transforms = "transforms"sv;
    // String
    SText envlight = "envlight"sv;
} // namespace dmt::dict

namespace dmt::parse_helpers {
    using json = nlohmann::json;

    enum class ExtractVec3fResult
    {
        eOk = 0,
        eNotArray,
        eIncorrectSize,
        eInvalidType0,
        eInvalidType1,
        eInvalidType2,

        eCount
    };

    enum class TempTexObjResult
    {
        eOk = 0,
        eNotExists,
        eFormatNotSupported,
        eFailLoad,
        eNumChannelsIncorrect,

        eCount
    };

    static constexpr bool hasOnlyAllowedkeys(json const& object, std::unordered_set<std::string> const& allowedKeys)
    {
        if (!object.is_object())
            return false;
        for (auto const& [key, val] : object.items())
        {
            if (!allowedKeys.contains(key))
                return false;
        }
        return true;
    }

    static constexpr bool hasAllAllowedkeys(json const& object, std::unordered_set<std::string> const& allowedKeys)
    {
        if (!hasOnlyAllowedkeys(object, allowedKeys))
            return false;
        for (auto const& key : allowedKeys)
        {
            if (!object.contains(key))
                return false;
        }
        return true;
    }

    static TempTexObjResult tempTexObj(os::Path const& path, bool isRGB, ImageTexturev2* out)
    {
        using enum TempTexObjResult;
        if (!path.isValid() || !path.isFile())
            return eNotExists;

        std::pmr::string           pathStr = path.toUnderlying();
        std::pmr::memory_resource* mem     = std::pmr::get_default_resource();

        int         xRes = 0, yRes = 0, channels = isRGB ? 3 : 1;
        void const* data = nullptr;
        TexFormat   format;
        if (pathStr.ends_with("png"))
        {
            format = isRGB ? TexFormat::ByteRGB : TexFormat::ByteGray;
            data   = stbi_load(pathStr.c_str(), &xRes, &yRes, &channels, 0);
            if (!data)
                return eFailLoad;
            if (channels != (isRGB ? 3 : 1))
                return eNumChannelsIncorrect;
        }
        else if (pathStr.ends_with("exr"))
        {
            try
            {
                Imf::RgbaInputFile file(pathStr.c_str());
                Imath::Box2i       dw = file.dataWindow();

                xRes = dw.max.x - dw.min.x + 1;
                yRes = dw.max.y - dw.min.y + 1;

                // Allocate OpenEXR storage
                Imf::Array2D<Imf::Rgba> pixels;
                pixels.resizeErase(yRes, xRes);

                file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * xRes, 1, xRes);
                file.readPixels(dw.min.y, dw.max.y);

                if (isRGB)
                {
                    format    = TexFormat::HalfRGB;
                    half* buf = (half*)malloc(sizeof(half) * xRes * yRes * 3);
                    if (!buf)
                        throw std::runtime_error("what");
                    for (int y = 0; y < yRes; ++y)
                    {
                        for (int x = 0; x < xRes; ++x)
                        {
                            Imf::Rgba const& px         = pixels[y][x];
                            buf[(y * xRes + x) * 3 + 0] = px.r;
                            buf[(y * xRes + x) * 3 + 1] = px.g;
                            buf[(y * xRes + x) * 3 + 2] = px.b;
                        }
                    }
                    data = buf;
                }
                else
                {
                    format    = TexFormat::HalfGray;
                    half* buf = (half*)malloc(sizeof(half) * xRes * yRes);
                    if (!buf)
                        throw std::runtime_error("what");
                    for (int y = 0; y < yRes; ++y)
                    {
                        for (int x = 0; x < xRes; ++x)
                        {
                            Imf::Rgba const& px = pixels[y][x];
                            // Simple luminance conversion (could be weighted differently)
                            buf[y * xRes + x] = half(0.299f * px.r + 0.587f * px.g + 0.114f * px.b);
                        }
                    }
                    data = buf;
                }
            } catch (std::exception const& e)
            {
                return eFailLoad;
            }
        }
        else
        {
            return eFormatNotSupported;
        }

        *out = makeRGBMipmappedTexture(data, xRes, yRes, TexWrapMode::eClamp, TexWrapMode::eClamp, format, mem);
        return eOk;
    }

    static void freeTempTexObj(os::Path const& path, ImageTexturev2& in)
    {
        std::pmr::string pathStr = path.toUnderlying();
        if (pathStr.ends_with("png"))
        {
            stbi_image_free(in.data);
        }
        else if (pathStr.ends_with("exr"))
        {
            free(in.data);
        }
    }

    static ExtractVec3fResult extractVec3f(json const& vec, Vector3f* out)
    {
        using enum ExtractVec3fResult;
        assert(out);
        if (!vec.is_array())
            return eNotArray;
        if (vec.size() != 3)
            return eIncorrectSize;
        for (uint32_t i = 0; i < 3; ++i)
        {
            try
            {
                if (!vec[i].is_number())
                    return static_cast<ExtractVec3fResult>(dmt::toUnderlying(eInvalidType0) + i);
                float elem = static_cast<float>(vec[i]);
                (*out)[i]  = elem;
            } catch (...)
            {
                return static_cast<ExtractVec3fResult>(dmt::toUnderlying(eInvalidType0) + i);
            }
        }

        return eOk;
    }

    static bool parseFilm(Parser& parser, json const& film, Parameters* param)
    {
        Context ctx;

        if (film.size() != 2)
        {
            //insert error message
            return false;
        }

        try
        {
            //keys check
            if (!film.contains("resolutionX"))
            {
                //insert error message
                return false;
            }

            if (!film.contains("resolutionY"))
            {
                //insert error message
                return false;
            }

            //Values check
            if (!film["resolutionX"].is_number())
            {
                //insert error message
                return false;
            }

            if (!film["resolutionY"].is_number())
            {
                //insert error message
                return false;
            }

            param->filmResolution = Point2i{film["resolutionX"], film["resolutionY"]};

        } catch (...)
        {
            //insert error message
            return false;
        }


        return true;
    }

    static bool parseTexture(Parser& parser, ParserState& state, json const& texture, Renderer* rend)
    {
        Context ctx;

        if (texture.size() != 3)
        {
            //insert error message
            return false;
        }

        if (!texture.contains("name"))
        {
            //insert error message
            return false;
        }

        if (!texture.contains("type"))
        {
            //insert error message
            return false;
        }

        if (!texture.contains("path"))
        {
            //insert error message
            return false;
        }

        if (state.texturesPath.find(texture["name"]) != state.texturesPath.end())
        {
            //insert error message
            return false;
        }

        if (texture["type"] != "diffuse" && texture["type"] != "normal" && texture["type"] != "metallic")
        {
            //insert error message
            return false;
        }

        std::string texName                 = texture["name"];
        state.texturesPath[texName].texType = texture["type"];
        os::Path texPath = parser.fileDirectory() / static_cast<std::string>(texture["path"]).c_str();

        if (!texPath.isValid() || !texPath.isFile())
        {
            //insert error message
            return false;
        }

        state.texturesPath[texName].texPath = std::string(texPath.toUnderlying());
        ImageTexturev2   imgTex;
        TempTexObjResult res = tempTexObj(texPath,
                                          state.texturesPath[texName].texType == "diffuse" ||
                                              state.texturesPath[texName].texType == "normal",
                                          &imgTex);
        if (res != TempTexObjResult::eOk)
        {
            ctx.error("Error loading texture", {});
            return false;
        }
        state.texturesPath[texName].xRes = static_cast<uint32_t>(imgTex.width);
        state.texturesPath[texName].yRes = static_cast<uint32_t>(imgTex.height);

        rend->texCache.MipcFiles.createCacheFile(baseKeyFromPath(texPath), imgTex);
        freeTempTexObj(texPath, imgTex);
        return true;
    }

    static bool parseMaterial(Parser& parser, ParserState& state, json const& material, Renderer* rend)
    {
        Context         ctx;
        SurfaceMaterial mat{};
        if (!material.is_object())
        {
            ctx.error("The 'materials' array should contain only objects", {});
            return false;
        }
        static std::unordered_set<std::string> const
            allowedKeys{"name",
                        "diffuse",
                        "metallic",
                        "normal",
                        "roughness",
                        "ior",
                        "eta",
                        "etak",
                        "ggx-anisotropy",
                        "ggx-dielectric",
                        "oren-nayar-dielectric"};
        if (!hasOnlyAllowedkeys(material, allowedKeys))
        {
            ctx.error("material has extraneous key", {});
            return false;
        }

        std::string name;
        try
        {
            if (material.contains("name"))
            {
                ctx.error(
                    "The material should have a unique 'name'"
                    "field in the materials namespace",
                    {});
                return false;
            }

            if (!state.materials.contains(material["name"]))
            {
                ctx.error("Duplicate material names", {});
                return false;
            }
            name = material["name"];
            // - mandatory fields parsing
            if (!material.contains("diffuse"))
            {
                ctx.error("material should specify a 'diffuse' either as RGB or texture name", {});
                return false;
            }

            if (!material.contains("metallic"))
            {
                ctx.error("material should specify a 'metallic' either as float or texture name", {});
                return false;
            }

            if (!material.contains("normal"))
            {
                ctx.error("material should specify a 'normal' either as RGB or texture name", {});
                return false;
            }

            if (!material.contains("roughness"))
            {
                ctx.error("material should specify a 'roughness' either as float or texture name", {});
                return false;
            }

            // -- diffuse
            if (material["diffuse"].is_array())
            {
                Vector3f tmp{};
                if (extractVec3f(material["diffuse"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("'diffuse' constant expected to be RGB value", {});
                    return false;
                }
                mat.diffusevalue = byte3FromRGB(RGB::fromVec(min(max(tmp, {0, 0, 0}), {1, 1, 1})));
            }
            else if (material["diffuse"].is_string())
            {
                std::string diffuseName = material["diffuse"];
                if (!state.texturesPath.contains(diffuseName))
                {
                    ctx.error("'diffuse' texture name should be an existing named texture", {});
                    return false;
                }

                TexInfo const& texInfo = state.texturesPath.at(diffuseName);
                if (texInfo.texType != "diffuse")
                {
                    ctx.error("'diffuse' material texture should point to a 'diffuse' texture", {});
                    return false;
                }
                mat.diffusekey    = baseKeyFromPath(texInfo.texPath);
                mat.diffuseWidth  = texInfo.xRes;
                mat.diffuseHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::DiffuseMask;
            }
            else
            {
                ctx.error("material 'diffuse' should be either texture name or RGB", {});
                return false;
            }

            // -- normal
            mat.useShadingNormals = true;
            if (material["normal"].is_array())
            {
                Vector3f tmp{};
                if (extractVec3f(material["normal"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("'normal' constant expected to be RGB value", {});
                }
                mat.normalvalue = octaFromNorm(normalize(tmp));
            }
            else if (material["normal"].is_string())
            {
                std::string normalName = material["normal"];
                if (!state.texturesPath.contains(normalName))
                {
                    ctx.error("'normal' texture name should be an existing named texture", {});
                    return false;
                }

                TexInfo const& texInfo = state.texturesPath.at(normalName);
                if (texInfo.texType != "normal")
                {
                    ctx.error("'normal' material texture should point to a 'normal' texture", {});
                    return false;
                }
                mat.normalkey    = baseKeyFromPath(texInfo.texPath);
                mat.normalWidth  = texInfo.xRes;
                mat.normalHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::NormalMask;
            }
            else
            {
                ctx.error("material 'normal' should be either texture name or RGB", {});
                return false;
            }

            // -- roughness
            if (material["roughness"].is_number())
            {
                mat.roughnessvalue = fl::clamp01(material["roughness"]);
            }
            else if (material["roughness"].is_string())
            {
                std::string roughnessName = material["roughness"];
                if (!state.texturesPath.contains(roughnessName))
                {
                    ctx.error("'roughness' texture name should be an existing named texture", {});
                    return false;
                }
                TexInfo const& texInfo = state.texturesPath.at(roughnessName);
                if (texInfo.texType != "roughness")
                {
                    ctx.error("'roughness' material texture should point to a 'roughness' texture", {});
                    return false;
                }
                mat.roughnesskey    = baseKeyFromPath(texInfo.texPath);
                mat.roughnessWidth  = texInfo.xRes;
                mat.roughnessHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::RoughnessMask;
            }
            else
            {
                ctx.error("material 'roughness' should be either texture name or float", {});
                return false;
            }

            // -- metallic
            if (material["metallic"].is_number())
            {
                mat.metallicvalue = fl::clamp01(material["metallic"]);
            }
            else if (material["metallic"].is_string())
            {
                std::string metallicName = material["metallic"];
                if (!state.texturesPath.contains(metallicName))
                {
                    ctx.error("'metallic' texture name should be an existing named texture", {});
                    return false;
                }
                TexInfo const& texInfo = state.texturesPath.at(metallicName);
                if (texInfo.texType != "metallic")
                {
                    ctx.error("'metallic' material texture should point to a 'metallic' texture", {});
                    return false;
                }
                mat.metallickey    = baseKeyFromPath(texInfo.texPath);
                mat.metallicWidth  = texInfo.xRes;
                mat.metallicHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::MetallicMask;
            }
            else
            {
                ctx.error("material 'metallic' should be either texture name or float", {});
                return false;
            }

            // - optional fields for diffuse
            // -- ior
            // dielectric ior list: <https://balyberdin.com/tools/ior-list/>
            mat.ior = 1.4f; // default
            if (material.contains("ior"))
            {
                if (!material["ior"].is_number())
                {
                    ctx.error("'ior' from material, if present, should be a number between 1 and 10", {});
                    return false;
                }
                mat.ior = fmaxf(material["ior"], 1.f);
            }

            // -- eta
            static constexpr uint8_t etaPresent  = 0x1;
            static constexpr uint8_t etakPresent = 0x2;
            static constexpr uint8_t etaFull     = etaPresent | etakPresent;
            uint8_t                  etaPresence = 0;
            // conductor ior list: <https://chris.hindefjord.se/resources/rgb-ior-metals/>
            mat.eta  = {0.18299, 0.42108, 1.37340};
            mat.etak = {3.42420, 2.34590, 1.77040};
            if (material.contains("eta"))
            {
                etaPresence |= etaPresent;
                Vector3f tmp{};
                if (extractVec3f(material["eta"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("material error extracting RGB 'eta' field", {});
                    return false;
                }
                mat.eta = RGB::fromVec(max(tmp, {0, 0, 0}));
            }

            // -- etak
            if (material.contains("etak"))
            {
                etaPresence |= etakPresent;
                Vector3f tmp{};
                if (extractVec3f(material["etak"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("material error extracting RGB 'etak' field", {});
                    return false;
                }
                mat.etak = RGB::fromVec(max(tmp, {0, 0, 0}));
            }

            if (etaPresence != 0 && etaPresence != etaFull)
            {
                ctx.error("material: either specify both 'eta' and 'etak' or none of them (Gold)", {});
                return false;
            }

            // -- ggx-anisotropy
            if (material.contains("ggx-anisotropy"))
            {
                if (!material["ggx-anisotropy"].is_number_float())
                {
                    ctx.error("material field 'ggx-anisotropy' should be a number from 0.0 to 1.0", {});
                    return false;
                }

                float const anisInterp = fl::clamp01(material["ggx-anisotropy"]);
                mat.anisotropy         = fl::lerp(anisInterp, 1.f, 8.f);
            }

            // -- ggx-dielectric
            bool dielectricFound = false;
            if (material.contains("ggx-dielectric"))
            {
                dielectricFound = true;

                json const& ggxDielectric = material["ggx-dielectric"];
                if (!ggxDielectric.is_object() && !ggxDielectric.is_null())
                {
                    ctx.error("material: 'ggx-dielectric' should be an object (or null to use defaults)", {});
                    return false;
                }
                mat.reflectanceTint   = {1, 1, 1};
                mat.transmittanceTint = {1, 1, 1};
                if (ggxDielectric.is_object())
                {
                    static std::unordered_set<std::string> const allowedKeys{"reflectance-tint", "transmittance-tint"};
                    if (!hasOnlyAllowedkeys(ggxDielectric, allowedKeys))
                    {
                        ctx.error("material: Unrecognized key inside 'ggx-dielectric'", {});
                        return false;
                    }
                    if (ggxDielectric.contains("reflectance-tint"))
                    {
                        Vector3f tmp{};
                        if (extractVec3f(ggxDielectric["reflectance-tint"], &tmp) != ExtractVec3fResult::eOk)
                        {
                            ctx.error("material: error extracting 'reflectance-tint'", {});
                            return false;
                        }
                        mat.reflectanceTint = RGB::fromVec(max(min(tmp, {1, 1, 1}), {0, 0, 0}));
                    }
                    if (ggxDielectric.contains("transmittance-tint"))
                    {
                        Vector3f tmp{};
                        if (extractVec3f(ggxDielectric["transmittance-tint"], &tmp) != ExtractVec3fResult::eOk)
                        {
                            ctx.error("material: error extracting 'transmittance-tint'", {});
                            return false;
                        }
                        mat.transmittanceTint = RGB::fromVec(min(max(tmp, {0, 0, 0}), {1, 1, 1}));
                    }
                }
            }

            // -- oren-nayar-dielectric
            if (material.contains("oren-nayar-dielectric"))
            {
                if (dielectricFound)
                {
                    ctx.error("material: Either specify 'ggx-dielectric' or 'oren-nayar-dielectric', not both", {});
                    return false;
                }
                dielectricFound                 = true;
                json const& orenNayarDielectric = material["oren-nayar-dielectric"];
                if (!orenNayarDielectric.is_object() && !orenNayarDielectric.is_null())
                {
                    ctx.error("material: 'oren-nayar-dielectric' should be an object (or null to use defaults)", {});
                    return false;
                }
                mat.multiscatterMultiplier = 1.f;
                if (orenNayarDielectric.contains("multiscatter-multiplier"))
                {
                    if (json const& j = orenNayarDielectric["multiscatter-multiplier"];
                        !j.is_number() || static_cast<float>(j) <= 0.f)
                    {
                        ctx.error("material: 'multiscatter-multiplier' should be a positive number", {});
                        return false;
                    }
                    mat.multiscatterMultiplier = orenNayarDielectric["multiscatter-multiplier"];
                }
            }
        } catch (...)
        {
            ctx.error("Unknown Error during material parsing", {});
            return false;
        }

        rend->scene.materials.push_back(mat);
        auto const& [it, wasInserted] = state.materials.try_emplace(name,
                                                                    static_cast<uint32_t>(rend->scene.materials.size() - 1));

        return wasInserted;
    }

    static bool parseCamera(Parser& parser, json const& camera, Parameters* param)
    {

        Context ctx;
        if (camera.size() != 3)
        {
            return false;
        }
        try
        {
            //check on key
            if (!camera.contains("focalLength"))
            {
                //insert error message
                return false;
            }

            if (!camera.contains("sensorSize"))
            {
                //insert error message
                return false;
            }

            if (!camera.contains("sensorSize"))
            {
                //insert error message
                return false;
            }

            if (!camera.contains("direction"))
            {
                //insert error message
                return false;
            }

            //check on type
            if (!camera["sensor"].is_number())
            {
                //insert error message
                return false;
            }

            if (!camera["focalLength"].is_number())
            {
                //insert error message
                return false;
            }

            json               jDir = camera["direction"];
            Vector3f           dir{};
            ExtractVec3fResult res = extractVec3f(jDir, &dir);

            if (res != ExtractVec3fResult::eOk)
            {
                //insert error message
                return false;
            }


            if (!camera["focalLength"].is_number())
            {
                //insert error message
                return false;
            }

            param->focalLength     = camera["focalLength"] / 1000.f;
            param->sensorSize      = camera["sensorSize"] / 1000.f;
            param->cameraDirection = dir;


        } catch (...)
        {
            //insert error message
            return false;
        }

        return true;
    }
    static bool transform(Parser& parser, ParserState& state, json const& transform)
    {
        Context ctx;
        // common: All transforms have a unique name: fail if name absent or duplicate
        // one of: "SRT"
        if (transform.size() != 2)
        {
            ctx.error("[Transform] unexpected number of elements", {});
            return false;
        }
        std::string name;
        json        srt;
        try
        {
            json jname = transform["name"];
            if (!jname.is_string() && (name = jname).empty())
            {
                ctx.error("Expected 'name' to be a non-empty string", {});
                return false;
            }
            srt = transform["srt"];
        } catch (...)
        {
            ctx.error("Expected 'name' and 'srt' as params for the transform object", {});
            return false;
        }

        // check existance
        if (state.transforms.contains(name))
        {
            ctx.error("Transfor with name {} already exists", std::make_tuple(name));
            return false;
        }

        // extract from srt values: "rotate-axis", "rotate-degrees", "translation-vector", "scale". If some of them
        // are absent, then we assume some defaults
        Vector3f rotateAxis    = {0, 0, 1};
        float    rotateDegrees = 0;
        Vector3f translationVector{};
        Vector3f scale{1, 1, 1}; // TODO should we allow small scales
        try
        {
            if (srt.contains("rotate-axis"))
            {
                Vector3f tmp{};
                if (extractVec3f(srt["rotate-axis"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("Error extracting field 'rotate-axis'", {});
                    return false;
                }
                rotateAxis = normalize(tmp);
            }
            if (srt.contains("rotate-degrees"))
            {
                if (!srt["rotate-degrees"].is_number())
                {
                    ctx.error("Error extracting field 'rotate-degrees'", {});
                    return false;
                }
                rotateDegrees = srt["rotate-degrees"];
            }
            if (srt.contains("translation-vector"))
            {
                Vector3f tmp{};
                if (extractVec3f(srt["translation-vector"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("Error extracting field 'translation-vector'", {});
                    return false;
                }
                translationVector = tmp;
            }
            if (srt.contains("scale"))
            {
                if (srt["scale"].is_number())
                    scale = Vector3f::s(static_cast<float>(srt["scale"]));
                else if (Vector3f tmp{}; extractVec3f(srt["scale"], &tmp) == ExtractVec3fResult::eOk)
                    scale = tmp;
                else
                {
                    ctx.error("Error parsing field 'scale'", {});
                    return false;
                }
            }
        } catch (...)
        {
            ctx.error("Unknown error during transform parsing", {});
            return false;
        }

        Transform t = Transform::translate(translationVector) * Transform::rotate(rotateDegrees, rotateAxis) *
                      Transform::scale(scale);
        auto const& [it, wasInserted] = state.transforms.try_emplace(name, t);

        return wasInserted;
    }

    static bool light(Parser& parser, ParserState& state, json const& light)
    {
        Context          ctx;
        std::string_view allowedKeysPoint[]{
            "name",
            "type",
            "cone-angle",
            "falloff-percentage",
            "radiant-intensity",
        };

        if (!light.is_object())
        {
            ctx.error("light is not an object", {});
            return false;
        }

        std::string name;
        std::string type;

        try
        {
            // compulsory name and type
            if (!light.contains("name") || !light["name"].is_string())
            {
                ctx.error("Absent or invalid type for light 'name'", {});
                return false;
            }
            name = light["name"];
            if (state.lights.contains(name))
            {
                ctx.error("Duplicated 'name' light found", {});
                return false;
            }

            if (!light.contains("type") || !light["type"].is_string())
            {
                ctx.error("Absent or invalid type for light 'type'", {});
                return false;
            }
            name = light["type"];

            if (type == "point") // if point, then only radiant-intensity is the other allowed field
            {
                if (light.size() > 3) // name, type, radiant-intensity
                {
                    ctx.error("Too many parameters for point light specification", {});
                    return false;
                }

                RGB radiantIntensity{1, 1, 1};
                if (light.contains("radiant-intensity"))
                {
                    Vector3f tmp{};
                    if (extractVec3f(light["radiant-intensity"], &tmp) != ExtractVec3fResult::eOk)
                    {
                        ctx.error("Incorrect format for 'radiant-intensity' of point light", {});
                        return false;
                    }
                    radiantIntensity = RGB::fromVec(tmp);
                }

                auto const& [it, wasInserted] = //
                    state.lights.try_emplace(name, makePointLight(Transform{}, radiantIntensity));
                if (!wasInserted)
                    return false;
            }
            else if (type == "spot") // if spot, then radiant-intensity, cone-angle, falloff-percentage are the only allowed field
            {
                if (light.size() > 4)
                {
                    ctx.error("Too many parameters for spot light specification", {});
                    return false;
                }

                RGB   radiantIntensity{1, 1, 1};
                float coneAngle         = 60.f;
                float falloffPercentage = 10.f;

                if (light.contains("radiant-intensity"))
                {
                    Vector3f tmp{};
                    if (extractVec3f(light["radiant-intensity"], &tmp) != ExtractVec3fResult::eOk)
                    {
                        ctx.error("Incorrect format for 'radiant-intensity' of point light", {});
                        return false;
                    }
                    radiantIntensity = RGB::fromVec(max(tmp, {0, 0, 0}));
                }

                if (light.contains("cone-angle"))
                {
                    if (!light["cone-angle"].is_number())
                    {
                        ctx.error("'cone-angle' should be a number", {});
                        return false;
                    }
                    coneAngle = fl::clamp(light["cone-angle"], 10.f, 120.f);
                }

                if (light.contains("falloff-percentage"))
                {
                    if (!light["falloff-percentage"].is_number())
                    {
                        ctx.error("'falloff-percentage', should be a number", {});
                        return false;
                    }
                    falloffPercentage = fl::clamp(light["falloff-percentage"], 1.f, 80.f);
                }

                float const cosTheta0         = cosf(coneAngle * (1.f - falloffPercentage / 100.f) * fl::pi() / 180.f);
                float const cosThetae         = cosf(coneAngle * fl::pi() / 180.f);
                auto const& [it, wasInserted] = //
                    state.lights.try_emplace(name, makeSpotLight(Transform{}, radiantIntensity, cosTheta0, cosThetae));
                if (!wasInserted)
                    return false;
            }
            else
            {
                ctx.error("Undefined 'type' for light", {});
                return false;
            }
        } catch (...)
        {
            ctx.error("Unknown error parsing light", {});
            return false;
        }

        return true;
    }

    static bool envlight(Parser& parser, ParserState& state, json const& envLight, Renderer* renderer)
    {
        Context ctx;
        if (!envLight.is_string())
        {
            ctx.error("expected 'envLight' to be string", {});
            return false;
        }

        os::Path envPath = parser.fileDirectory() / static_cast<std::string>(envLight).c_str();
        __debugbreak(); // TODO remove
        if (!envPath.isValid() || !envPath.isFile())
        {
            ctx.error(
                "invalid path specified on 'envlight'."
                " Should be a PNG/EXR file relative to JSON file directory",
                {});
        }
        assert(false); // TODO construct envlight in params
    }

    static bool object(Parser& parser, ParserState& state, json const& object, Renderer* rend, MeshFbxParser& meshParser)
    {
        Context ctx;

        if (!object.is_object())
        {
            ctx.error("element of 'objects' should be a JSON object", {});
            return false;
        }

        std::string name;
        // grows depending on type. once type fully specified, check
        std::unordered_set<std::string> allowedKeys{"name", "type", "material"};
        allowedKeys.reserve(128);
        try
        {
            if (!object.contains("name") || !object["name"].is_string())
            {
                ctx.error("object specification mandates a unique 'name' as string", {});
                return false;
            }
            name = object["name"];
            if (state.objects.contains(name))
            {
                ctx.error("object 'name's should be unique", {});
                return false;
            }

            // get material id
            if (!object.contains("material") || !object["material"].is_string())
            {
                ctx.error("object should specify a string 'material' name", {});
                return false;
            }

            std::string const materialStr = object["material"];
            if (!state.materials.contains(materialStr))
            {
                ctx.error("couldn't find material of name `{}`", std::make_tuple(materialStr));
                return false;
            }
            uint32_t const materialIdx = state.materials.at(materialStr);

            // switch on type
            if (!object.contains("type") || !object["type"].is_string())
            {
                ctx.error("object should contain a string 'type'", {});
                return false;
            }
            std::string const type = object["type"];

            rend->scene.geometry.emplace_back(makeUniqueRef<TriangleMesh>(rend->scene.memory()));
            TriangleMesh& mesh = *rend->scene.geometry.back();
            if (type == "fbx" || type == "FBX")
            {
                allowedKeys.insert("path");
                if (!hasOnlyAllowedkeys(object, allowedKeys))
                {
                    ctx.error("Unrecognized keys in object", {});
                    return false;
                }
                // check path is valid and file, then go to fbx parsing function
                if (!object.contains("path") || !object["path"].is_string())
                {
                    ctx.error("FBX object should contain a string 'path' field", {});
                    return false;
                }
                os::Path const fbxPath = parser.fileDirectory() / static_cast<std::string>(object["path"]).c_str();
                if (!fbxPath.isValid() || !fbxPath.isFile())
                {
                    ctx.error("FBX object 'path' field should point to a existing FBX file. Got `{}`",
                              std::make_tuple(fbxPath.toUnderlying()));
                    return false;
                }

                if (!meshParser.ImportFBX(fbxPath.toUnderlying().c_str(), &mesh))
                {
                    __debugbreak(); // TODO remove
                    ctx.error("FBX: Error during parsing of FBX file", {});
                    return false;
                }

                for (size_t tri = 0; tri < mesh.triCount(); ++tri)
                {
                    mesh.getIndexedTriRef(tri).matIdx = materialIdx;
                }
            }
            else if (type == "primitive")
            {
                allowedKeys.insert("shape");
                if (!hasOnlyAllowedkeys(object, allowedKeys))
                {
                    ctx.error("Unrecognized keys in object", {});
                    return false;
                }

                if (!object.contains("shape") || !object["shape"].is_string())
                {
                    ctx.error("'shape' for primitive absent or ill-defined", {});
                    return false;
                }

                std::string const shape = object["shape"];
                if (shape == "cube")
                    TriangleMesh::unitCube(mesh, materialIdx);
                else if (shape == "plane")
                    TriangleMesh::unitPlane(mesh, materialIdx);
                else
                {
                    ctx.error("Unrecognized 'shape' `{}`", std::make_tuple(shape));
                    return false;
                }
            }
            else
            {
                ctx.error("Unrecognized type `{}` in object parsing", std::make_tuple(type));
                return false;
            }
        } catch (...)
        {
            ctx.error("Unknown error while parsing object", {});
            return false;
        }

        auto const& [it, wasInserted] = //
            state.objects.try_emplace(name, static_cast<uint32_t>(rend->scene.geometry.size() - 1));

        return wasInserted;
    }
} // namespace dmt::parse_helpers

namespace dmt {
    Parser::Parser(os::Path const& path, Renderer* renderer, std::pmr::memory_resource* mem) :
    m_renderer{renderer},
    m_path{path},
    m_tmp(mem)
    {
    }

    bool Parser::parse()
    {
        using json = nlohmann::json;
        Context ctx;
        if (!ctx.isValid())
            return false;

        std::ifstream f(m_path.toUnderlying(m_tmp).c_str());
        if (!f)
            return false;

        static std::unordered_set<std::string> const
            allowedKeys{"camera",
                        "film",
                        "textures",
                        "materials",
                        "objects",
                        "lights",
                        "envlight",
                        "transforms",
                        "world"};

        try
        {
            json data = json::parse(f);
            if (!parse_helpers::hasAllAllowedkeys(data, allowedKeys))
                throw std::runtime_error("JSON value is lacking some keys");

            ParserState state;

            // parse camera object
            if (!parse_helpers::parseCamera(*this, data["camera"], &m_renderer->params))
                throw nullptr;

            // parse film object
            if (!parse_helpers::parseFilm(*this, data["film"], &m_renderer->params))
                throw nullptr;

            // parse texture array
            if (!data["textures"].is_array())
                throw std::runtime_error("'textures' should be a JSON array");

            for (json const& texture : data["textures"])
            {
                if (!parse_helpers::parseTexture(*this, state, texture, m_renderer))
                    throw nullptr;
            }

            // parse materials array
            if (!data["materials"].is_array())
                throw std::runtime_error("'materials' should be a JSON array");

            for (json const& material : data["materials"])
            {
                if (!parse_helpers::parseMaterial(*this, state, material, m_renderer))
                    throw nullptr;
            }

            // parse objects array
            if (!data["objects"].is_array())
                throw std::runtime_error("'objects' should be a JSON array");

            for (json const& object : data["objects"])
            {
                if (!parse_helpers::object(*this, state, object, m_renderer, m_fbxParser))
                    throw nullptr;
            }

            // parse envlight
            if (!parse_helpers::envlight(*this, state, data["envlight"], m_renderer))
                throw nullptr;

            // parse transforms array
            if (!data["transforms"].is_array())
                throw std::runtime_error("'transforms' should be a JSON array");

            for (json const& transform : data["transforms"])
            {
                if (!parse_helpers::transform(*this, state, transform))
                    throw nullptr;
            }

            // parse world
            assert(false); // TODO
        } catch (...)
        {
            try
            {
                auto excPtr = std::current_exception();
                if (excPtr)
                    std::rethrow_exception(excPtr);
            } catch (std::exception const& e)
            {
                ctx.error("[Parser JSON Exception] {}", std::make_tuple(e.what()));
            }
        }
        // true if successful
        return false;
    }
} // namespace dmt