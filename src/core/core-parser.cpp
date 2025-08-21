#include "core-parser.h"

#include "platform/platform-context.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <unordered_map>

namespace dmt {
    struct ParserState
    {
        std::unordered_map<std::string, Transform> transforms;
        std::unordered_map<std::string, Light>     lights;
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

    static bool camera(Parser& parser, json const& camera, Parameters* param)
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

            param->focalLength     = camera["focalLength"];
            param->sensorSize      = camera["sensorSize"];
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
        std::string_view allowedKeys[]{
            "type",
            "cone-angle",
            "falloff-percentage",
            "intensity",
        };

        try
        {
        } catch (...)
        {
            ctx.error("Unknown error parsing light", {});
            return false;
        }
    }
} // namespace dmt::parse_helpers

namespace dmt {
    Parser::Parser(os::Path const& path, Parameters* pParameters, Scene* pScene, std::pmr::memory_resource* mem) :
    m_pScene(pScene),
    m_parameters{pParameters},
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

        try
        {
            json data = json::parse(f);
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