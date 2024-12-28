module;

#include "dmtmacros.h"

#include <array>
#include <string_view>
#include <type_traits>

#include <cstdint>
#include <cstring>

module middleware;

template <typename Enum, size_t N>
    requires(std::is_enum_v<Enum>)
static constexpr Enum enumFromStr(char const* str, std::array<std::string_view, N> const& types, Enum defaultEnum)
{
    for (uint8_t i = 0; i < types.size(); ++i)
    {
        if (std::strncmp(str, types[i].data(), types[i].size()) == 0)
        {
            return ::dmt::fromUnderlying<Enum>(i);
        }
    }
    return defaultEnum;
}

namespace dmt {
    ERenderCoordSys renderCoordSysFromStr(char const* str)
    { // array needs to follow the order in which the enum values are declared
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ERenderCoordSys> count = toUnderlying(ERenderCoordSys::eCount);
        static constexpr std::array<std::string_view, count>     types{"cameraworld"sv, "camera"sv, "world"sv};

        return ::enumFromStr(str, types, ERenderCoordSys::eCameraWorld);
    }

    ECameraType cameraTypeFromStr(char const* str)
    { // array needs to follow the order in which the enum values are declared
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ECameraType> count = toUnderlying(ECameraType::eCount);
        static constexpr std::array<std::string_view, count>
            types{"orthographic"sv, "perspective"sv, "realistic"sv, "spherical"sv};

        return ::enumFromStr(str, types, ECameraType::ePerspective);
    }

    ESphericalMapping sphericalMappingFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ESphericalMapping> count = toUnderlying(ESphericalMapping::eCount);
        static constexpr std::array<std::string_view, count>       types{"equalarea"sv, "equirectangular"sv};

        return ::enumFromStr(str, types, ESphericalMapping::eEqualArea);
    }

    ESamplerType samplerTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ESamplerType> count = toUnderlying(ESamplerType::eCount);
        static constexpr std::array<std::string_view, count>
            types{"halton"sv, "independent"sv, "paddedsobol"sv, "sobol"sv, "stratified"sv, "zsobol"sv};

        return ::enumFromStr(str, types, ESamplerType::eZSobol);
    }

    ERandomization randomizationFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ERandomization> count = toUnderlying(ERandomization::eCount);
        static constexpr std::array<std::string_view, count>
            types{"fastowen"sv, "none"sv, "permutedigits"sv, "owen"sv};

        return ::enumFromStr(str, types, ERandomization::eFastOwen);
    }

    EColorSpaceType colorSpaceTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EColorSpaceType> count = toUnderlying(EColorSpaceType::eCount);
        static constexpr std::array<std::string_view, count> types{"srgb"sv, "rec2020"sv, "aces2065-1"sv, "dci-p3"sv};

        return ::enumFromStr(str, types, EColorSpaceType::eSRGB);
    }

    EFilmType filmTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EFilmType>   count = toUnderlying(EFilmType::eCount);
        static constexpr std::array<std::string_view, count> types{"rgb"sv, "gbuffer"sv, "spectral"sv};

        return ::enumFromStr(str, types, EFilmType::eRGB);
    }

    ESensor sensorFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ESensor> count = toUnderlying(ESensor::eCount);
        static constexpr std::array<std::string_view, count>
            types{"cie1931"sv,
                  "canon_eos_100d"sv,
                  "canon_eos_1dx_mkii"sv,
                  "canon_eos_200d"sv,
                  "canon_eos_200d_mkii"sv,
                  "canon_eos_5d"sv,
                  "canon_eos_5d_mkii"sv,
                  "canon_eos_5d_mkiii"sv,
                  "canon_eos_5d_mkiv"sv,
                  "canon_eos_5ds"sv,
                  "canon_eos_m"sv,
                  "hasselblad_l1d_20c"sv,
                  "nikon_d810"sv,
                  "nikon_d850"sv,
                  "sony_ilce_6400"sv,
                  "sony_ilce_7m3"sv,
                  "sony_ilce_7rm3"sv,
                  "sony_ilce_9"sv};

        return ::enumFromStr(str, types, ESensor::eCIE1931);
    }

    EGVufferCoordSys gBufferCoordSysFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EGVufferCoordSys> count = toUnderlying(EGVufferCoordSys::eCount);
        static constexpr std::array<std::string_view, count>      types{"camera"sv, "world"sv};

        return ::enumFromStr(str, types, EGVufferCoordSys::eCamera);
    }
} // namespace dmt

namespace dmt::model {
} // namespace dmt::model

namespace dmt::job {
}