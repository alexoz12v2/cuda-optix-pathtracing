#pragma once

#include "dmtmacros.h"

#include <middleware/middleware-utils.h>

#if !defined(DMT_NEEDS_MODULE)
#include <platform/platform.h>

#include <array>
#include <atomic>
#include <forward_list>
#include <limits>
#include <map>
#include <memory_resource>
#include <set>
#include <stack>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>
#include <compare>
#include <cstdint>
#endif

DMT_MODULE_EXPORT namespace dmt {
    enum class DMT_MIDDLEWARE_API ERenderCoordSys : uint8_t
    {
        eCameraWorld = 0,
        eCamera,
        eWorld,
        eCount
    };

    // f = .pbrt file option, c = command line option, o = refers to the presence or absence of some other option
    // some file options are also present from command line and override what's in the file
    // the files ones are to OR with what we got from the command line, hence the job will yield bool options too
    enum class DMT_MIDDLEWARE_API EBoolOptions : uint32_t
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
        ecWavefront             = 1u << 8u, // cam be set by file too
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
        return static_cast<EBoolOptions>(toUnderlying(lhs) | toUnderlying(rhs));
    }

    inline constexpr EBoolOptions operator&(EBoolOptions lhs, EBoolOptions rhs)
    {
        return static_cast<EBoolOptions>(toUnderlying(lhs) & toUnderlying(rhs));
    }

    inline constexpr EBoolOptions operator^(EBoolOptions lhs, EBoolOptions rhs)
    {
        return static_cast<EBoolOptions>(toUnderlying(lhs) ^ toUnderlying(rhs));
    }

    inline constexpr EBoolOptions operator~(EBoolOptions val) { return static_cast<EBoolOptions>(~toUnderlying(val)); }

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

    inline constexpr bool hasFlag(EBoolOptions value, EBoolOptions flag) { return (value & flag) == flag; }

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
        char* imageFile;
        char* mseReferenceImage;
        char* mseReferenceOutput;
        char* debugStart;
        char* displayServer;

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

    inline bool wavefrontOrGPU(Options const& options)
    {
        return hasFlag(options.flags, EBoolOptions::ecUseGPU | EBoolOptions::ecWavefront);
    }

    // Camera ---------------------------------------------------------------------------------------------------------
    // https://pbrt.org/fileformat-v4#cameras
    enum class DMT_MIDDLEWARE_API ECameraType : uint8_t
    {
        eOrthographic = 0,
        ePerspective,
        eRealistic,
        eSpherical,
        eCount
    };

    enum class DMT_MIDDLEWARE_API ESphericalMapping : uint8_t
    {
        eEqualArea = 0,
        eEquirectangular,
        eCount
    };

    namespace apertures {
        using namespace std::string_view_literals;
        inline constexpr std::string_view gaussian = "gaussian"sv;
        inline constexpr std::string_view square   = "square"sv;
        inline constexpr std::string_view pentagon = "pentagon"sv;
        inline constexpr std::string_view star     = "star"sv;
    } // namespace apertures

    struct CameraSpec
    {
    public:
        static constexpr float invalidScreen      = -std::numeric_limits<float>::infinity();
        static constexpr float invalidAspectRatio = -1.f;
        CameraSpec()                              = default;
        // Since realistic camrea stores two filenames as strings, we cannot use memcpy for copy semantics
        DMT_MIDDLEWARE_API             CameraSpec(CameraSpec const&);
        DMT_MIDDLEWARE_API             CameraSpec(CameraSpec&&) noexcept;
        DMT_MIDDLEWARE_API CameraSpec& operator=(CameraSpec const&);
        DMT_MIDDLEWARE_API CameraSpec& operator=(CameraSpec&&) noexcept;
        DMT_MIDDLEWARE_API ~CameraSpec() noexcept;

    public:
        struct Projecting // perspective or orthographic
        {                 // params are all lowercase in the file
            struct ScreenWindow
            {
                float minX;
                float maxX;
                float minY;
                float maxY;
            };
            union UScreenWindow
            {
                UScreenWindow()
                { // set to an arbitrary invalid value
                    float val = invalidScreen;
                    for (auto& v : arr)
                        v = val;
                }
                ScreenWindow         p;
                std::array<float, 4> arr;
            };
            float frameAspectRatio = invalidAspectRatio; // computed from film
            // [-1, 1] along shorter axis, [-screnWindow, +screenWindow] in longer axis. Default = aspect ratio if > 1, otherwise 1/aspect ratio
            UScreenWindow screenWindow;
            float         lensRadius    = 0;
            float         focalDistance = 1e30f; // 10^30 is near the float limit
            float         fov           = 90.f;  // Used only by perspective
        };
        struct Spherical
        {
            ESphericalMapping mapping;
        };
        struct Realistic
        {
            std::string lensfile;
            std::string aperture; // either a starndar one or a file
            float       apertureDiameter = 1.f;
            float       focusDistance    = 10.f;
        };

        union Params
        {
            Params() : p({}) {}
            ~Params() {}
            Realistic  r;
            Projecting p;
            Spherical  s;
        };

    private:
        void assignParams(Params const& that, bool move);

    public:
        // 8 Byte aligned
        Params params;

        // 4 byte aligned
        float shutteropen  = 0;
        float shutterclose = 1;

        // 1 byte aligned
        ECameraType type = ECameraType::ePerspective;
    };

    // Samplers -------------------------------------------------------------------------------------------------------
    enum class DMT_MIDDLEWARE_API ESamplerType : uint8_t
    {
        eZSobol = 0,
        eHalton,
        eIndependent,
        ePaddedSobol,
        eSobol,
        eStratified,
        eCount
    };

    enum class DMT_MIDDLEWARE_API ERandomization : uint8_t
    {
        eFastOwen = 0,
        eNone,
        ePermuteDigits,
        eOwen,
        eCount
    };

    struct DMT_MIDDLEWARE_API SamplerSpec
    {
        struct DMT_MIDDLEWARE_API StratifiedSamples
        {
            int32_t x      = 4;
            int32_t y      = 4;
            bool    jitter = true;
        };

        // everyone
        int32_t seed; // default from options, file takes precedence

        union DMT_MIDDLEWARE_API Samples
        {
            Samples() : num(16) {}

            // StatifiedSampler only
            StratifiedSamples stratified;
            // everyone but StratifiedSampler
            int32_t num;
        };
        Samples samples;

        // Low discrepancy sampleers (halton, padded sobol, sobol, zsobol)
        ERandomization randomization = ERandomization::eFastOwen;

        // everyone
        ESamplerType type = ESamplerType::eZSobol;
    };

    // Color Spaces ---------------------------------------------------------------------------------------------------
    enum class DMT_MIDDLEWARE_API EColorSpaceType : uint8_t
    {
        eSRGB = 0,
        eRec2020,
        eAces2065_1,
        eDci_p3,
        eCount
    };

    struct DMT_MIDDLEWARE_API ColorSpaceSpec
    {
        EColorSpaceType type = EColorSpaceType::eSRGB;
    };

    // Film -----------------------------------------------------------------------------------------------------------
    enum class DMT_MIDDLEWARE_API EFilmType : uint8_t
    {
        eRGB = 0,
        eGBuffer,
        eSpectral,
        eCount
    };

    enum class DMT_MIDDLEWARE_API ESensor : uint8_t
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

    enum class DMT_MIDDLEWARE_API EGVufferCoordSys : uint8_t
    {
        eCamera = 0,
        eWorld,
        eCount
    };

    struct FilmSpec
    {
        // common params
        std::string fileName = "pbrt.exr";

        int32_t                xResolution = 1280;
        int32_t                yResolution = 720;
        std::array<float, 4>   cropWindow{0.f, 1.f, 0.f, 1.f}; // TODO change type
        std::array<int32_t, 4> pixelBounds{0, xResolution, 0, yResolution};

        float diagonal          = 35.f; // mm
        float iso               = 100.f;
        float whiteBalance      = 0.f;
        float maxComponentValue = std::numeric_limits<float>::infinity();

        // spectral only
        int16_t nBuckets  = 16;
        float   lambdaMin = 360.f;
        float   lambdaMax = 830.f;

        // gbuffer only
        EGVufferCoordSys coordSys = EGVufferCoordSys::eCamera;

        // common params
        bool      savefp16 = true;
        ESensor   sensor   = ESensor::eCIE1931;
        EFilmType type     = EFilmType::eRGB;
    };

    // Filters --------------------------------------------------------------------------------------------------------
    enum class DMT_MIDDLEWARE_API EFilterType : uint8_t
    {
        eGaussian = 0,
        eBox,
        eMitchell,
        eSinc,
        eTriangle,
        eCount
    };

    float                     defaultRadiusFromFilterType(EFilterType e);
    struct DMT_MIDDLEWARE_API FilterSpec
    {
        struct DMT_MIDDLEWARE_API Gaussian
        {
            float sigma = 0.5f;
        };
        struct DMT_MIDDLEWARE_API Mitchell
        {
            static constexpr float oneThird = 0x1.3333333p-2f;

            float b = oneThird;
            float c = oneThird;
        };
        struct DMT_MIDDLEWARE_API Sinc
        {
            float tau = 3.f;
        };
        union DMT_MIDDLEWARE_API Params
        {
            Params() : gaussian({}) {}

            Gaussian gaussian;
            Mitchell mitchell;
            Sinc     sinc;
        };

        Params      params;
        float       xRadius = defaultRadiusFromFilterType(EFilterType::eGaussian);
        float       yRadius = defaultRadiusFromFilterType(EFilterType::eGaussian);
        EFilterType type    = EFilterType::eGaussian;
    };

    // Integrators ----------------------------------------------------------------------------------------------------
    // default is volpath, but if --gpu or --wavefront are specified, the type from file is ignored and set to
    // wavefront or gpu
    enum class DMT_MIDDLEWARE_API EIntegratorType : uint8_t
    {
        eVolPath = 0,
        eAmbientOcclusion,
        eBdpt,
        eLightPath,
        eMLT,
        ePath,
        eRandomWalk,
        eSimplePath,
        eSimpleVolPath,
        eSPPM, // stochastic progressive photon mapping
        eCount
    };

    enum class DMT_MIDDLEWARE_API ELightSampler : uint8_t
    {
        eBVH = 0,
        eUniform,
        ePower,
        eCount
    };

    struct DMT_MIDDLEWARE_API IntegratorSpec
    {
        struct DMT_MIDDLEWARE_API AmbientOcclusion
        {
            float maxDistance = std::numeric_limits<float>::infinity();
            bool  cosSample   = true;
        };
        struct DMT_MIDDLEWARE_API MetropolisTransport
        {
            float   sigma                  = 0.01f;
            float   largestStepProbability = 0.3f;
            int32_t mutationsPerPixel      = 100;
            int32_t chains                 = 1000;
            int32_t bootstraqpSamples      = 100000;
        };
        struct DMT_MIDDLEWARE_API BiDirPathTracing
        {
            bool visualizeStrategies = false;
            bool visualizeWeights    = false;
        };
        struct DMT_MIDDLEWARE_API SimplePath
        {
            bool sampleBSDF   = true;
            bool sampleLights = true;
        };
        struct DMT_MIDDLEWARE_API StocProgPhotMap
        {
            int32_t photonsPerIteration = -1;
            float   radius              = 0;
            int32_t seed                = 0;
        };
        union DMT_MIDDLEWARE_API Params
        {
            Params() {}

            AmbientOcclusion    ao;
            BiDirPathTracing    bdpt;
            MetropolisTransport mlt;
            SimplePath          simplePath;
            StocProgPhotMap     sppm;
        };
        Params params;

        int32_t         maxDepth     = 5; // all but ambient occlusion
        bool            regularize   = false;
        ELightSampler   lightSampler = ELightSampler::eBVH; // path, volpath, gpu, wavefront
        EIntegratorType type         = EIntegratorType::eVolPath;
    };

    // Acceletators ---------------------------------------------------------------------------------------------------
    enum class DMT_MIDDLEWARE_API EAcceletatorType : uint8_t
    {
        eBVH = 0,
        eKdTree,
        eCount
    };

    enum class DMT_MIDDLEWARE_API EBVHSplitMethod : uint8_t
    {
        eSAH = 0,
        eMiddle,
        eEqual,
        eHLBVH,
        eCount
    };

    struct DMT_MIDDLEWARE_API AcceleratorSpec
    {
        struct DMT_MIDDLEWARE_API BVH
        {
            int32_t         maxNodePrims = 4;
            EBVHSplitMethod splitMethod  = EBVHSplitMethod::eSAH;
        };
        struct DMT_MIDDLEWARE_API KDTree
        {
            int32_t intersectCost = 5;
            int32_t traversalCost = 1;
            float   emptyBonus    = 0.5f;
            int32_t maxPrims      = 1;
            int32_t maxDepth      = -1;
        };
        union DMT_MIDDLEWARE_API Params
        {
            Params() : bvh({}) {}

            BVH    bvh;
            KDTree kdtree;
        };

        Params           params;
        EAcceletatorType type = EAcceletatorType::eBVH;
    };

    // WorldBegin -----------------------------------------------------------------------------------------------------
    // Participating Media -------------------------------------------------------------
    // present both in global fragment (where the camera ray starts in) and after the `WorldBegin`
    // Parsing --------------------------------------------------------------------------------------------------------

    /**
     * If this gives problems of stack overflow, either allocate it on the stack allocator, or let this class take
     * the memory context and allocate the buffer somewhere else
     */
    class DMT_MIDDLEWARE_API WordParser
    {
    public:
        /**
         * number of max ASCII characters in a token (concert only for names and comments, which are the
         * longset tokens)
         */
        static constexpr uint32_t maxTokenSize = 256;
        using CharBuffer                       = std::array<char, maxTokenSize>;

        std::string_view nextWord(std::string_view str);
        bool             needsContinuation() const;
        uint32_t         numCharReadLast() const;

    private:
        char             decodeEscaped(char c);
        char             getChar(std::string_view str, size_t idx);
        void             copyToBuffer(std::string_view str);
        std::string_view catResult(std::string_view str, size_t start, size_t end);
        bool             endOfStr(std::string_view str, size_t idx) const;

        uint32_t m_bufferLength        = 0;
        uint32_t m_numCharReadLastTime = 0;
        char     m_buffer[maxTokenSize]{};
        char     m_escapedBuffer[maxTokenSize]{};
        bool     m_needsContinuation = false;
        bool     m_haveEscaped       = false;
    };

    // ----------------------------------------------------------------------------------------------------------------
    using ArgsDArray = std::vector<std::string>;
    struct ParamPair
    {
        using ValueList = ArgsDArray;
        constexpr explicit ParamPair(sid_t type) : type(type) {}

        DMT_MIDDLEWARE_API void addParamValue(std::string_view value) { values.emplace_back(value); }

        DMT_MIDDLEWARE_API std::string_view valueAt(uint32_t i) const { return values[i]; }

        DMT_MIDDLEWARE_API uint32_t numParams() const { return static_cast<uint32_t>(values.size()); }

        // TODO better (char buffer fixed)
        ArgsDArray values;

        sid_t type;
    };

    using ParamMap = std::map<sid_t, ParamPair>;

    enum class DMT_MIDDLEWARE_API ETarget : uint8_t
    {
        eShape = 0,
        eLight,
        eMaterial,
        eMedium,
        eTexture,
        eCount
    };

    enum class DMT_MIDDLEWARE_API ETextureType : uint8_t
    {
        eSpectrum = 0,
        eFloat
    };

    enum class DMT_MIDDLEWARE_API ETextureClass : uint8_t
    {
        eBilerp = 0,
        eCheckerboard,
        eConstant,
        eDirectionmix,
        eDots,
        eFbm,
        eImagemap,
        eMarble,
        eMix,
        ePtex,
        eScale,
        eWindy,
        eWrinkled,
        eCount
    };

    enum class DMT_MIDDLEWARE_API EMaterialType : uint8_t
    {
        eCoateddiffuse = 0,
        eCoatedconductor,
        eConductor,
        eDielectric,
        eDiffuse,
        eDiffusetransmission,
        eHair,
        eInterface,
        eMeasured,
        eMix,
        eSubsurface,
        eThindielectric,
        eCount
    };

    enum class DMT_MIDDLEWARE_API ELightType : uint8_t
    {
        eDistant = 0,
        eGoniometric,
        eInfinite,
        ePoint,
        eProjection,
        eSpot,
        eCount
    };

    enum class DMT_MIDDLEWARE_API EAreaLightType : uint8_t
    {
        eDiffuse = 0,
        eCount
    };

    enum class DMT_MIDDLEWARE_API EShapeType : uint8_t
    {
        eBilinearmesh = 0,
        eCurve,
        eCylinder,
        eDisk,
        eSphere,
        eTrianglemesh,
        eLoopsubdiv,
        ePlymesh,
        eCount
    };

    enum class DMT_MIDDLEWARE_API EActiveTransform : uint8_t
    {
        eStartTime = 0,
        eEndTime,
        eAll,
        eCount
    };

    struct DMT_MIDDLEWARE_API SceneEntity
    {
        // SceneEntity Public Methods
        SceneEntity() = default;
        SceneEntity(sid_t& name, ParamMap parameters) : name(sid_t), parameters(parameters) {}
    }

    // SceneEntity Public Members
    sid_t    name;
    ParamMap parameters;
    };

    // CameraSceneEntity Definition
    struct DMT_MIDDLEWARE_API CameraSceneEntity : public SceneEntity {
    // CameraSceneEntity Public Methods
    CameraSceneEntity() = default;
    CameraSceneEntity(const sid_t& name, CameraSpec parameters, const CameraTransform&, const sid_t& medium) :
    name(name),
    params(parameters),
    cameraTransform(cameraTransform),
    medium(medium)
    {
    }

    sid_t           name;
    sid_t           medium;
    CameraTransform cameraTransform;
    CameraSpec      params;
    };


    // LightSource ----------------------------------------------------------------------------------------------------
    struct DMT_MIDDLEWARE_API LightSourceSpec
    {
    // 4 byte aligned, coommon
    union DMT_MIDDLEWARE_API PowerOrIlluminance
    { // there is no default, one of these must be present
        // all except distant and infinite
        float power;
        // distant, infinite
        float illuminance;
        bool  illum;
    };
    PowerOrIlluminance po;
    float              scale = 1.f;
    // 1 byte aligned
    ELightType type;
    };

    // Parsing --------------------------------------------------------------------------------------------------------
    class TokenStream
    {
public:
    static constexpr uint32_t chunkSize = 512;

    DMT_MIDDLEWARE_API TokenStream(AppContext & actx, std::string_view filePath);
    TokenStream(TokenStream const&)                = delete;
    TokenStream(TokenStream&&) noexcept            = delete;
    TokenStream& operator=(TokenStream const&)     = delete;
    TokenStream& operator=(TokenStream&&) noexcept = delete;
    DMT_MIDDLEWARE_API ~TokenStream() noexcept;

    DMT_MIDDLEWARE_API std::string next(AppContext & actx);
    DMT_MIDDLEWARE_API void        advance(AppContext & actx);
    DMT_MIDDLEWARE_API std::string peek();

private:
    union U
    {
        U() {}
        ~U() {}

        ChunkedFileReader reader;
    };
    U                m_delayedCtor;
    WordParser*      m_tokenizer = nullptr;
    std::string      m_token;
    std::string_view m_chunk       = "";
    char*            m_buffer      = nullptr;
    uint32_t         m_chunkNum    = 0;
    bool             m_newChunk    = true;
    bool             m_needAdvance = true;
    };

    struct DMT_MIDDLEWARE_API EndOfHeaderInfo
    {
    CameraSpec cameraSpec;
    };

    class DMT_MIDDLEWARE_API DMT_INTERFACE IParserTarget
    {
public:
    virtual void Scale(float sx, float sy, float sz) = 0;

    virtual void Shape(EShapeType type, ParamMap const& params) = 0;

    virtual ~IParserTarget(){};

    virtual void Option(sid_t name, ParamPair const& value) = 0;

    virtual void Identity()                                                                                       = 0;
    virtual void Translate(float dx, float dy, float dz)                                                          = 0;
    virtual void Rotate(float angle, float ax, float ay, float az)                                                = 0;
    virtual void LookAt(float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz) = 0;
    virtual void ConcatTransform(std::array<float, 16> const& transform)                                          = 0;
    virtual void Transform(std::array<float, 16> transform)                                                       = 0;
    virtual void CoordinateSystem(sid_t name)                                                                     = 0;
    virtual void CoordSysTransform(sid_t name)                                                                    = 0;
    virtual void ActiveTransformAll()                                                                             = 0;
    virtual void ActiveTransformEndTime()                                                                         = 0;
    virtual void ActiveTransformStartTime()                                                                       = 0;
    virtual void TransformTimes(float start, float end)                                                           = 0;

    virtual void ColorSpace(EColorSpaceType colorSpace)   = 0;
    virtual void PixelFilter(FilterSpec const& spec)      = 0;
    virtual void Film(FilmSpec const& spec)               = 0;
    virtual void Accelerator(AcceleratorSpec const& spec) = 0;
    virtual void Integrator(IntegratorSpec const& spec)   = 0;
    virtual void Camera(CameraSpec const& params)         = 0;
    virtual void Sampler(SamplerSpec const& spec)         = 0;

    virtual void MakeNamedMedium(sid_t name, ParamMap const& params)  = 0;
    virtual void MediumInterface(sid_t insideName, sid_t outsideName) = 0;

    virtual void WorldBegin()                                                                          = 0;
    virtual void AttributeBegin()                                                                      = 0;
    virtual void AttributeEnd()                                                                        = 0;
    virtual void Attribute(ETarget target, ParamMap const& params)                                     = 0;
    virtual void Texture(sid_t name, ETextureType type, ETextureClass texname, ParamMap const& params) = 0;
    virtual void Material(EMaterialType type, ParamMap const& params)                                  = 0;
    virtual void MakeNamedMaterial(sid_t name, ParamMap const& params)                                 = 0;
    virtual void NamedMaterial(sid_t name)                                                             = 0;
    virtual void LightSource(ELightType type, ParamMap const& params)                                  = 0;
    virtual void AreaLightSource(EAreaLightType type, ParamMap const& params)                          = 0;
    virtual void ReverseOrientation()                                                                  = 0;
    virtual void ObjectBegin(sid_t name)                                                               = 0;
    virtual void ObjectEnd()                                                                           = 0;
    virtual void ObjectInstance(sid_t name)                                                            = 0;

    virtual void EndOfHeader(EndOfHeaderInfo const& info) = 0;
    virtual void EndOfOptions(Options const& options)     = 0;
    virtual void EndOfFiles()                             = 0;
    };

    // TODO move
    struct DMT_MIDDLEWARE_API TransformSet
    {
    static constexpr uint32_t maxTransforms = 2;
    // TransformSet Public Methods
    Transform& operator[](int i)
    {
        assert(i >= 0);
        assert(i < maxTransforms);
        return t[i];
    }
    Transform const& operator[](int i) const
    {
        assert(i >= 0);
        assert(i < maxTransforms);
        return t[i];
    }

    friend TransformSet Inverse(TransformSet const& ts)
    {
        TransformSet tInv = ts;
        for (int i = 0; i < maxTransforms; ++i)
            tInv.t[i].inverse();
        return tInv;
    }

    bool IsAnimated() const
    {
        for (int i = 0; i < maxTransforms - 1; ++i)
            if (t[i] != t[i + 1])
                return true;
        return false;
    }

private:
    Transform t[maxTransforms];
    };

    struct GraphicsState
    {
public:
    template <typename F>
        requires std::is_invocable_r_v<dmt::Transform, F, dmt::Transform>
    void ForActiveTransforms(F func)
    {
        for (int i = 0; i < TransformSet::maxTransforms; ++i)
            if (activeTransformBits & (1 << i))
                ctm[i] = func(ctm[i]);
    }

public:
    sid_t currentInsideMedium  = 0;
    sid_t currentOutsideMedium = 0;

    sid_t   currentMaterialName  = 0;
    int32_t currentMaterialIndex = 0;

    sid_t    areaLightName = 0;
    ParamMap areaLightParams;

    ParamMap shapeAttributes;
    ParamMap lightAttributes;
    ParamMap materialAttributes;
    ParamMap textureAttributes;

    TransformSet ctm;
    uint32_t     activeTransformBits = std::numeric_limits<uint32_t>::max();
    bool         reverseOrientation  = false;
    float        transformStartTime  = 0.f;
    float        transformEndTime    = 1.f;


    std::map<sid_t, TransformSet> namedCoordinateSystems;
    EColorSpaceType               colorSpace = EColorSpaceType::eSRGB;
    };

    class SceneDescription : public IParserTarget
    {
public:
    DMT_MIDDLEWARE_API void Scale(float sx, float sy, float sz) override;
    DMT_MIDDLEWARE_API void Shape(EShapeType type, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void Option(sid_t name, ParamPair const& value) override;
    DMT_MIDDLEWARE_API void Identity() override;
    DMT_MIDDLEWARE_API void Translate(float dx, float dy, float dz) override;
    DMT_MIDDLEWARE_API void Rotate(float angle, float ax, float ay, float az) override;
    DMT_MIDDLEWARE_API void LookAt(float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz)
        override;
    DMT_MIDDLEWARE_API void ConcatTransform(std::array<float, 16> const& transform) override;
    DMT_MIDDLEWARE_API void Transform(std::array<float, 16> transform) override;
    DMT_MIDDLEWARE_API void CoordinateSystem(sid_t name) override;
    DMT_MIDDLEWARE_API void CoordSysTransform(sid_t name) override;
    DMT_MIDDLEWARE_API void ActiveTransformAll() override;
    DMT_MIDDLEWARE_API void ActiveTransformEndTime() override;
    DMT_MIDDLEWARE_API void ActiveTransformStartTime() override;
    DMT_MIDDLEWARE_API void TransformTimes(float start, float end) override;
    DMT_MIDDLEWARE_API void ColorSpace(EColorSpaceType colorSpace) override;
    DMT_MIDDLEWARE_API void PixelFilter(FilterSpec const& spec) override;
    DMT_MIDDLEWARE_API void Film(FilmSpec const& spec) override;
    DMT_MIDDLEWARE_API void Accelerator(AcceleratorSpec const& spec) override;
    DMT_MIDDLEWARE_API void Integrator(IntegratorSpec const& spec) override;
    DMT_MIDDLEWARE_API void Camera(CameraSpec const& params) override;
    DMT_MIDDLEWARE_API void MakeNamedMedium(sid_t name, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void MediumInterface(sid_t insideName, sid_t outsideName) override;
    DMT_MIDDLEWARE_API void Sampler(SamplerSpec const& spec) override;
    DMT_MIDDLEWARE_API void WorldBegin() override;
    DMT_MIDDLEWARE_API void AttributeBegin() override;
    DMT_MIDDLEWARE_API void AttributeEnd() override;
    DMT_MIDDLEWARE_API void Attribute(ETarget target, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void Texture(sid_t name, ETextureType type, ETextureClass texname, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void Material(EMaterialType type, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void MakeNamedMaterial(sid_t name, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void NamedMaterial(sid_t name) override;
    DMT_MIDDLEWARE_API void LightSource(ELightType type, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void AreaLightSource(EAreaLightType type, ParamMap const& params) override;
    DMT_MIDDLEWARE_API void ReverseOrientation() override;
    DMT_MIDDLEWARE_API void ObjectBegin(sid_t name) override;
    DMT_MIDDLEWARE_API void ObjectEnd() override;
    DMT_MIDDLEWARE_API void ObjectInstance(sid_t name) override;
    DMT_MIDDLEWARE_API void EndOfOptions(Options const& options) override;
    DMT_MIDDLEWARE_API void EndOfFiles() override;
    DMT_MIDDLEWARE_API void EndOfHeader(EndOfHeaderInfo const& info) override;

public:
    CameraSpec      cameraSpec;
    SamplerSpec     samplerSpec;
    ColorSpaceSpec  colorSpaceSpec;
    FilmSpec        filmSpec;
    FilterSpec      filterSpec;
    IntegratorSpec  integratorSpec;
    AcceleratorSpec acceleratorSpec;

private:
    GraphicsState     graphicsState;
    CameraSceneEntity camera;
    };

    enum class DMT_MIDDLEWARE_API EParsingStep : uint8_t
    {
        eOptions = 0,
        eHeader,
        eWorld
    };

    enum class DMT_MIDDLEWARE_API EScope : uint8_t
    {
        eAttribute = 0,
        eObject
    };

    enum class DMT_MIDDLEWARE_API EEncounteredHeaderDirective : uint32_t
    {
        eNone        = 0u,
        eCamera      = 1u,
        eSampler     = 1u << 1u,
        eColorSpace  = 1u << 2u,
        eFilm        = 1u << 3u,
        eIntegrator  = 1u << 4u,
        ePixelFilter = 1u << 5u,
        eAccelerator = 1u << 6u,
    };
    inline constexpr bool hasFlag(EEncounteredHeaderDirective e, EEncounteredHeaderDirective val)
    {
    return (toUnderlying(e) & toUnderlying(val)) != 0;
    }
    inline EEncounteredHeaderDirective putFlag(EEncounteredHeaderDirective e, EEncounteredHeaderDirective val)
    {
    return static_cast<EEncounteredHeaderDirective>(toUnderlying(e) | toUnderlying(val));
    }
    // TODO move to cpp with dict constants
    inline std::string_view toStr(EEncounteredHeaderDirective e)
    {
    using namespace std::string_view_literals;
    switch (e)
    {
        using enum EEncounteredHeaderDirective;
        case eCamera: return "Camera"sv;
        case eSampler: return "Sampler"sv;
        case eColorSpace: return "ColorSpace"sv;
        case eFilm: return "Film"sv;
        case eIntegrator: return "Integrator"sv;
        case ePixelFilter: return "PixelFilter"sv;
        case eAccelerator: return "Accelerator"sv;
    }
    return ""sv;
    }

    // TOSO Option directives must precede anything else
    // TODO AreaLight is applied to a shape AFTERWARDS
    class SceneParser
    {
public:
    DMT_MIDDLEWARE_API      SceneParser(AppContext & actx, IParserTarget * pTarget, std::string_view filePath);
    DMT_MIDDLEWARE_API void parse(AppContext & actx, Options & inOutOptions);

private:
    static bool setOptionParam(AppContext & actx, ParamMap const& params, Options& outOptions);

private:
    uint32_t parseArgs(AppContext & actx, TokenStream & stream, ArgsDArray & outArr);
    uint32_t parseParams(AppContext & actx, TokenStream & stream, ParamMap & outParams);
    bool transitionToHeaderIfFirstHeaderDirective(AppContext & actx, Options const& outOptions, EEncounteredHeaderDirective val);
    void         pushFile(AppContext & actx, std::string_view filePath, bool isImportOrMainFile);
    void         popFile(bool isImportOrMainFile);
    TokenStream& topFile();
    bool         hasScope() const;
    EScope       currentScope() const;
    void         popScope();
    void         pushScope(EScope scope);

private:
    struct ParsingState
    {
        // populated once the Camera directive is encountered (or left default constructed)
        CameraSpec cameraSpec;
        // populated once the Film directive is encountered
        int32_t xResolution = -1;
        int32_t yResolution = -1;
    };

    // TODO better
    std::forward_list<TokenStream>   m_fileStack;
    std::vector<std::vector<EScope>> m_scopeStacks;
    std::string                      m_basePath;
    IParserTarget*                   m_pTarget;

    ParsingState m_parsingState;

    EParsingStep                m_parsingStep        = EParsingStep::eOptions;
    EEncounteredHeaderDirective m_encounteredHeaders = EEncounteredHeaderDirective::eNone;
    };
} // namespace dmt