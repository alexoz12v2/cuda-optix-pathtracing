#pragma once

#include "dmtmacros.h"

#include <array>
#include <atomic>
#include <limits>
#include <string_view>

#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
#include <platform/platform.h>
#else
import platform;
#endif

// stuff related to .pbrt file parsing
DMT_MODULE_EXPORT dmt {
    enum class ERenderCoordSys : uint8_t
    {
        eCameraWorld = 0,
        eCamera,
        eWorld,
        eCount
    };

    ERenderCoordSys renderCoordSysFromStr(char const* str);

    // f = .pbrt file option, c = command line option, o = refers to the presence or absence of some other option
    // some file options are also present from command line and override what's in the file
    // the files ones are to OR with what we got from the command line, hence the job will yield bool options too
    enum class EBoolOptions : uint32_t
    {
        eEmpty = 0,

        efDisablepixeljitter      = 1u << 0u,
        efDisabletexturefiltering = 1u << 1u,
        efDisablewavelengthjitter = 1u << 2u,
        efForcediffuse            = 1u << 3u,
        efPixelstats              = 1u << 4u,
        efWavefront               = 1u << 5u,

        ecDisableImageTextures  = 1u << 6u,
        ecUseGPU                = 1u << 7u,
        ecWavefront             = 1u << 8u,
        ecInteractive           = 1u << 9u,
        ecFullscreen            = 1u << 10u,
        ecLogUtilization        = 1u << 11u,
        ecWritePartialImages    = 1u << 12u,
        ecRecordPixelStatistics = 1u << 13u,
        ecPrintStatistics       = 1u << 14u,
        ecQuickRender           = 1u << 15u,
        ecUpgrade               = 1u << 16u,

        eoPixelSamples  = 1u << 17u,
        eoGpuDevice     = 1u << 18u,
        eoCropWindow    = 1u << 19u,
        eoPixelBounds   = 1u << 20u,
        eoPixelMaterial = 1u << 21u,
    };

    inline constexpr EBoolOptions operator|(EBoolOptions lhs, EBoolOptions rhs)
    {
        return static_cast<EBoolOptions>(static_cast<uint16_t>(lhs) | static_cast<uint16_t>(rhs));
    }

    inline constexpr EBoolOptions operator&(EBoolOptions lhs, EBoolOptions rhs)
    {
        return static_cast<EBoolOptions>(static_cast<uint16_t>(lhs) & static_cast<uint16_t>(rhs));
    }

    inline constexpr EBoolOptions operator^(EBoolOptions lhs, EBoolOptions rhs)
    {
        return static_cast<EBoolOptions>(static_cast<uint16_t>(lhs) ^ static_cast<uint16_t>(rhs));
    }

    inline constexpr EBoolOptions operator~(EBoolOptions val)
    {
        return static_cast<EBoolOptions>(~static_cast<uint16_t>(val));
    }

    inline constexpr EBoolOptions& operator|=(EBoolOptions& lhs, EBoolOptions rhs)
    {
        lhs = lhs | rhs;
        return lhs;
    }

    inline constexpr EBoolOptions& operator&=(EBoolOptions& lhs, EBoolOptions rhs)
    {
        lhs = lhs & rhs;
        return lhs;
    }

    inline constexpr EBoolOptions& operator^=(EBoolOptions& lhs, EBoolOptions rhs)
    {
        lhs = lhs ^ rhs;
        return lhs;
    }

    inline constexpr bool hasFlag(EBoolOptions value, EBoolOptions flag)
    {
        return (value & flag) == flag;
    }

    /**
     * Rendering options parsed from command line and from the input file
     * It's a subset of what pbrt-v4 supports. Reference links:
     * - [PBRT config source](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/options.h)
     * - [PBRT File Format](https://pbrt.org/fileformat-v4#scene-wide-rendering-options)
     * of these, only flags, seed, render space are needed by GPU
     */
    struct Options
    {
        // 8 byte aligned (not using string view as it uses a size_t as length, too much)
        char const* imageFile;
        char const* mseReferenceImage;
        char const* mseReferenceOutput;
        char const* debugStart;
        char const* displayServer;

        // 4 byte aligned
        uint32_t imageFileLength          = 0;
        uint32_t mseReferenceImageLength  = 0;
        uint32_t mseReferenceOutputLength = 0;
        uint32_t debugStartLength         = 0;
        uint32_t displayServerLength      = 0;

        EBoolOptions flags                 = EBoolOptions::eEmpty;
        uint32_t     numThreads            = 0;
        float        diespacementEdgeScale = 1;
        int32_t      seed                  = 0;

        // see flags to check whether these si valid or not
        uint32_t               pixelSamples = 0;
        uint32_t               gpuDevice    = 0;
        std::array<float, 2>   cropWindow{};    // TODO change type
        std::array<int32_t, 2> pixelBounds{};   // TODO change type
        std::array<int32_t, 2> pixelMaterial{}; // TODO change type

        // 1 byte aligned
        ELogLevel       logLevel    = ELogLevel::ERR;
        ERenderCoordSys renderCoord = ERenderCoordSys::eCameraWorld;
        unsigned char   padding[16];
    };
    static_assert(std::is_trivially_destructible_v<Options> && std::is_standard_layout_v<Options>);
    static_assert(sizeof(Options) == 128 && alignof(Options) == 8);

    // Camera ---------------------------------------------------------------------------------------------------------
    // https://pbrt.org/fileformat-v4#cameras
    enum class ECameraType : uint8_t
    {
        eOrthographic = 0,
        ePerspective,
        eRealistic,
        eSpherical,
        eCount
    };

    ECameraType cameraTypeFromStr(char const* str);

    enum class ESphericalMapping : uint8_t
    {
        eEqualArea = 0,
        eEquirectangular,
        eCount
    };

    ESphericalMapping sphericalMappingFromStr(char const* str);

    namespace apertures {
        using namespace std::string_view_literals;
        static inline constexpr std::string_view gaussian = "gaussian"sv;
        static inline constexpr std::string_view square   = "square"sv;
        static inline constexpr std::string_view pentagon = "pentagon"sv;
        static inline constexpr std::string_view star     = "star"sv;
    }

    struct CameraSpec
    {
        struct Projecting           // perspective or orthographic
        {                           // params are all lowercase in the file
            float frameAspectRatio; // computed from film
            float screenWindow;     // [-1, 1] along shorter axis, [-screnWindow, +screenWindow] in longer axis
            float lensRadius    = 0;
            float focalDistance = 1e30f; // 10^30 is near the float limit
            float fov           = 90.f;  // Used only by perspective
        };
        struct Spherical
        {
            ESphericalMapping mapping;
        };
        struct Realistic
        {
            std::string_view lensfile;
            std::string_view aperture; // either a starndar one or a file
            float            apertureDiameter = 1.f;
            float            focusDistance    = 10.f;
        };

        union Params
        {
            Realistic  r;
            Projecting p;
            Spherical  s;
        };

        // 8 Byte aligned
        Params params;

        // 4 byte aligned
        float shutteropen  = 0;
        float shutterclose = 1;

        // 1 byte aligned
        ECameraType type;
    };

    // Samplers -------------------------------------------------------------------------------------------------------
    enum class ESamplerType : uint8_t
    {
        eZSobol = 0,
        eHalton,
        eIndependent,
        ePaddedSobol,
        eSobol,
        eStratified,
        eCount
    };

    ESamplerType samplerTypeFromStr(char const* str);

    enum class ERandomization : uint8_t
    {
        eFastOwen = 0,
        eNone,
        ePermuteDigits,
        eOwen,
        eCount
    };

    ERandomization randomizationFromStr(char const* str);

    struct SamplerSpec
    {
        struct StratifiedSamples
        {
            int32_t x = 4;
            int32_t y = 4;
        };

        // everyone
        int32_t seed; // default from options, file takes precedence

        union Samples
        {
            // StatifiedSampler only
            StratifiedSamples stratified;
            // everyone but StratifiedSampler
            int32_t num = 16;
        };
        Samples samples;

        // Low discrepancy sampleers (halton, padded sobol, sobol, zsobol)
        ERandomization randomization;

        // everyone
        ESamplerType type;
    };

    // Color Spaces ---------------------------------------------------------------------------------------------------
    enum class EColorSpaceType : uint8_t
    {
        eSRGB = 0,
        eRec2020,
        eAces2065_1,
        eDci_p3,
        eCount
    };

    EColorSpaceType colorSpaceTypeFromStr(char const* str);

    struct ColorSpaceSpec
    {
        EColorSpaceType type;
    };

    // Film -----------------------------------------------------------------------------------------------------------
    enum class EFilmType : uint8_t
    {
        eRGB = 0,
        eGBuffer,
        eSpectral,
        eCount
    };

    EFilmType filmTypeFromStr(char const* str);

    enum class ESensor : uint8_t
    {
        eCIE1931 = 0,
        eCanon_eos_100d,
        eCanon_eos_1dx_mkii,
        eCanon_eos_200d,
        eCanon_eos_200d_mkii,
        eCanon_eos_5d,
        eCanon_eos_5d_mkii,
        eCanon_eos_5d_mkiii,
        eCanon_eos_5d_mkiv,
        eCanon_eos_5ds,
        eCanon_eos_m,
        eHasselblad_l1d_20c,
        eNikon_d810,
        eNikon_d850,
        eSony_ilce_6400,
        eSony_ilce_7m3,
        eSony_ilce_7rm3,
        eSony_ilce_9,
        eCount
    };

    ESensor sensorFromStr(char const* str);

    enum class EGVufferCoordSys : uint8_t
    {
        eCamera = 0,
        eWorld,
        eCount
    };

    EGVufferCoordSys gBufferCoordSysFromStr(char const* str);

    struct FilmSpec
    {
        // common params
        std::string_view fileName = "pbrt.exr";

        uint32_t xResolution = 1280;
        uint32_t yResolution = 720;
        std::array<float, 4> cropWindow{0.f, 1.f, 0.f, 1.f}; // TODO change type
        std::array<uint32_t, 4> pixelBounds{0u, xResolution, 0u, yResolution};

        float diagonal = 35.f; // mm
        float iso      = 100.f;
        float whiteBalance = 0.f;
        float maxComponentValue = std::numeric_limits<float>::infinity();

        // spectral only
        uint16_t nBuckets = 16;
        float    lambdaMin = 360.f;
        float    lambdaMax = 830.f;

        // gbuffer only
        EGVufferCoordSys coordSys;

        // common params
        bool      savefp16 = true;
        ESensor   sensor = ESensor::eCIE1931;
        EFilmType type = EFilmType::eRGB;
    };

    // Filters --------------------------------------------------------------------------------------------------------
    // Integrators ----------------------------------------------------------------------------------------------------
    // Acceletators ---------------------------------------------------------------------------------------------------
    // Participating Media --------------------------------------------------------------------------------------------
}

DMT_MODULE_EXPORT dmt::model {}

DMT_MODULE_EXPORT dmt::job {
    struct ParseSceneHeaderData
    {
        std::atomic<uint32_t> done;
        dmt::AppContext&      actx;
        Options               outOptions;
        uint32_t              numChunkWorldBegin;
        uint32_t              offsetWorldBegin;
    };
}