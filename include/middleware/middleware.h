#pragma once

#include "dmtmacros.h"

#include <array>
#include <atomic>
#include <limits>
#include <string_view>
#include <thread>

#include <cassert>
#include <compare>
#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
#include <platform/platform.h>
#else
import platform;
#endif

// stuff related to .pbrt file parsing + data structures
DMT_MODULE_EXPORT dmt {
    inline size_t alignedSize(size_t elementSize, size_t alignment)
    {
        assert(alignment > 0 && "Alignment must be greater than 0");
        assert((alignment & (alignment - 1)) == 0 && "Alignment must be a power of 2");

        // Compute the aligned size of each element
        size_t alignedSize = (elementSize + alignment - 1) & ~(alignment - 1);
        return alignedSize;
    }

    inline size_t computeMaxElements(size_t bufferSize, size_t elementSize, size_t alignment)
    {
        size_t alignedSz = alignedSize(elementSize, alignment);

        // Return the number of elements that fit in the buffer
        return bufferSize / alignedSz;
    }

    struct AllocatorTable
    {
        static AllocatorTable fromPool(MemoryContext& mctx);

        TaggedPointer (*allocate)(MemoryContext& mctx, size_t size, size_t alignment);
        void (*free)(MemoryContext& mctx, TaggedPointer pt, size_t size, size_t alignment);
        void* (*rawPtr)(TaggedPointer pt);
    };

    AllocatorTable AllocatorTable::fromPool(MemoryContext & mctx)
    {
        AllocatorTable table;
        table.allocate = [](MemoryContext& mctx, size_t size, size_t alignment) {
            uint32_t numBlocks = static_cast<uint32_t>(ceilDiv(size, static_cast<size_t>(toUnderlying(EBlockSize::e32B))));
            return mctx.poolAllocateBlocks(numBlocks, EBlockSize::e32B, EMemoryTag::eUnknown, 0);
        };
        table.free = [](MemoryContext& mctx, TaggedPointer pt, size_t size, size_t alignment) {
            uint32_t numBlocks = static_cast<uint32_t>(ceilDiv(size, static_cast<size_t>(toUnderlying(EBlockSize::e32B))));
            mctx.poolFreeBlocks(numBlocks, pt);
        };
        table.rawPtr = [](TaggedPointer pt) { return pt.pointer(); };
        return table;
    }

    // ----------------------------------------------------------------------------------------------------------------

    struct Bitmap
    {
        Bitmap() : map(0)
        {
        }

        std::atomic<uint64_t> map;

        uint64_t getValue(uint64_t index) const
        {
            assert(index < 32);
            uint64_t const shamt = index << 1;
            uint64_t const mask  = 0b11ULL << shamt;

            uint64_t mapVal = map.load(std::memory_order_relaxed);
            uint64_t bits   = (mapVal & mask) >> shamt;
            return bits;
        }

        bool setValue(uint64_t value, uint64_t index)
        {
            return cas(value, index, false, 0, 0);
        }

    protected:
        bool checkValue(uint64_t value, uint64_t index) const
        {
            assert((value & ~0b11ULL) == 0);
            return getValue(index) == value;
        }

        void waitForValue(uint64_t value, uint64_t index) const
        {
            assert(index < 32);
            assert((value & ~0b11ULL) == 0);
            uint64_t const shamt  = index << 1;
            uint64_t const mask   = 0b11ULL << shamt;
            uint64_t       mapVal = map.load(std::memory_order_relaxed);
            uint64_t       bits   = mapVal & mask;
            while (bits != value)
            {
                std::this_thread::yield();
                mapVal = map.load(std::memory_order_relaxed);
                bits   = mapVal & mask;
            }
        }

        bool cas(uint64_t value, uint64_t index, bool checkFor, uint64_t expected0, uint64_t expected1)
        {
            assert(index < 32);
            assert((value & ~0b11ULL) == 0);
            assert((expected0 & ~0b11ULL) == 0);
            assert((expected1 & ~0b11ULL) == 0);

            uint64_t const shamt = index << 1;       // Shift amount for 2-bit slots
            uint64_t const mask  = 0b11ULL << shamt; // Mask for the target slot

            uint64_t mapVal = map.load(std::memory_order_relaxed);
            uint64_t bits   = (mapVal & mask) >> shamt; // Extract the bits for the given index

            if (checkFor)
            {
                // If checkFor is true, the value at the index must match one of the expected values
                if (bits != expected0 && bits != expected1)
                {
                    return false; // Value doesn't match either expected value
                }
            }

            // Perform the unconditional update if checkFor is false, or update after match if true
            uint64_t const desired = (mapVal & ~mask) | (value << shamt);
            return map.compare_exchange_strong(mapVal, desired, std::memory_order_seq_cst, std::memory_order_relaxed);
        }
    };

    struct SNodeBitmap : public Bitmap
    {
        static constexpr uint64_t free        = 0b00;
        static constexpr uint64_t readLocked  = 0b01;
        static constexpr uint64_t writeLocked = 0b10;
        static constexpr uint64_t unused      = 0b11;

        // clang-format off
        bool checkFree(uint64_t index) const { return checkValue(free, index); }
        bool checkReadLocked(uint64_t index) const { return checkValue(readLocked, index); }
        bool checkWriteLocked(uint64_t index) const { return checkValue(writeLocked, index); }
        bool checkUnused(uint64_t index) const { return checkValue(unused, index); }

        bool setFree(uint64_t index) { return cas(free, index, false, 0, 0); }
        bool casToReadLocked(uint64_t index) { return cas(readLocked, index, true, unused, readLocked); }
        bool casFreeToWriteLocked(uint64_t index, uint64_t exp = free, uint64_t exp2 = free) { return cas(writeLocked, index, true, exp, exp2); }
        bool setUnused(uint64_t index) { return cas(unused, index, false, 0, 0); }
        // clang-format on
    };

    struct INodeBitmap : public Bitmap
    {
        static constexpr uint64_t free     = 0b00;
        static constexpr uint64_t inode    = 0b01;
        static constexpr uint64_t snode    = 0b10;
        static constexpr uint64_t occupied = 0b11;

        // clang-format off
        bool checkFree(uint64_t index) const { return checkValue(free, index); }
        bool checkINode(uint64_t index) const { return checkValue(inode, index); }
        bool checkSNode(uint64_t index) const { return checkValue(snode, index); }
        bool checkOccupied(uint64_t index) const { return checkValue(occupied, index); }
        
        bool setOccupied(uint64_t index, bool checkTag) { return cas(occupied, index, true, inode, snode); }
        bool setOccupied(uint64_t index) { return cas(occupied, index, false, 0, 0); }
        bool setSNode(uint64_t index) { return cas(snode, index, false, 0, 0); }
        bool setINode(uint64_t index) { return cas(inode, index, false, 0, 0); }
        bool setFree(uint64_t index) { return cas(free, index, false, 0, 0); }
        // clang-format on
    };

    inline constexpr uint32_t branchFactorINode = 5;
    inline constexpr uint32_t cardinalityINode  = 1u << 5;

    struct SNode
    {
    public:
        bool                   tryReadLock(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        bool                   releaseReadLock(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        uint32_t               incrRefCounter(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        uint32_t               decrRefCounter(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        uint32_t&              keyRef(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        uint32_t               keyCopy(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        void const*            valueConstAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign) const;
        void*                  valueAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign);
        std::atomic<uint32_t>& refCounterAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign);

    private:
        uintptr_t getElementBaseAddress(uint32_t index, uint32_t valueSize, uint32_t valueAlign) const;


    public:
        using Data = std::array<unsigned char, 256>;
        alignas(8) std::array<unsigned char, 256> data;
        SNodeBitmap bits;
    };

    struct INode
    {
    public:
        std::array<TaggedPointer, cardinalityINode> children;
        INodeBitmap                                 bits;
    };
    static_assert(sizeof(SNode) == sizeof(INode));

    class CTrie
    {
    public:
        CTrie(MemoryContext& mctx, AllocatorTable const& table, uint32_t valueSize, uint32_t valueAlign);
        void cleanup(MemoryContext& mctx, void (*dctor)(MemoryContext& mctx, void* value));

        bool insert(MemoryContext& mctx, uint32_t key, void const* value);
        //bool remove(MemoryContext& mctx, uint32_t key);

    private:
        void cleanupINode(MemoryContext& mctx, INode* inode, void (*dctor)(MemoryContext& mctx, void* value));
        void cleanupSNode(MemoryContext& mctx, SNode* inode, void (*dctor)(MemoryContext& mctx, void* value));
        //bool iinsert(MemoryContext& mctx, TaggedPointer node, uint32_t key, void const* value, uint32_t level, INode* parent);
        bool reinsert(MemoryContext& mctx, INode* inode, uint32_t key, void const* value, uint32_t level);

        TaggedPointer newINode(MemoryContext& mctx) const;
        TaggedPointer newSNode(MemoryContext& mctx) const;

    private:
        AllocatorTable m_table;
        TaggedPointer  m_root;
        uint32_t       m_valueSize;
        uint32_t       m_valueAlign;
    };


    // ----------------------------------------------------------------------------------------------------------------

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

        uint32_t                xResolution = 1280;
        uint32_t                yResolution = 720;
        std::array<float, 4>    cropWindow{0.f, 1.f, 0.f, 1.f}; // TODO change type
        std::array<uint32_t, 4> pixelBounds{0u, xResolution, 0u, yResolution};

        float diagonal          = 35.f; // mm
        float iso               = 100.f;
        float whiteBalance      = 0.f;
        float maxComponentValue = std::numeric_limits<float>::infinity();

        // spectral only
        uint16_t nBuckets  = 16;
        float    lambdaMin = 360.f;
        float    lambdaMax = 830.f;

        // gbuffer only
        EGVufferCoordSys coordSys;

        // common params
        bool      savefp16 = true;
        ESensor   sensor   = ESensor::eCIE1931;
        EFilmType type     = EFilmType::eRGB;
    };

    // Filters --------------------------------------------------------------------------------------------------------
    enum class EFilterType : uint8_t
    {
        eGaussian = 0,
        eBox,
        eMitchell,
        eSinc,
        eTriangle,
        eCount
    };

    struct FilterSpec
    {
        struct Gaussian
        {
            float sigma = 0.5f;
        };
        struct Mitchell
        {
            static constexpr float oneThird = 0x1.3333333p-2f;

            float b = oneThird;
            float c = oneThird;
        };
        struct Sinc
        {
            float tau = 3.f;
        };
        union Params
        {
            Gaussian gaussian;
            Mitchell mitchell;
            Sinc     sinc;
        };

        Params      params;
        float       xRadius;
        float       yRadius;
        EFilterType type;
    };

    EFilterType filterTypeFromStr(char const* str);
    float       defaultRadiusFromFilterType(EFilterType e);

    // Integrators ----------------------------------------------------------------------------------------------------
    // default is volpath, but if --gpu or --wavefront are specified, the type from file is ignored and set to
    // wavefront or gpu
    enum class EIntegratorType : uint8_t
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

    EIntegratorType integratorTypeFromStr(char const* str);

    enum class ELightSampler : uint8_t
    {
        eBVH = 0,
        eUniform,
        ePower,
        eCount
    };

    ELightSampler lightSamplerFromStr(char const* str);

    struct IntegratorSpec
    {
        struct AmbientOcclusion
        {
            float maxDistance = std::numeric_limits<float>::infinity();
            bool  cosSample   = true;
        };
        struct BiDirPathTracing
        {
            float    sigma                  = 0.01f;
            float    largestStepProbability = 0.3f;
            uint32_t mutationsPerPixel      = 100u;
            uint32_t chains                 = 1000u;
            uint32_t bootstraqpSamples      = 100000u;
        };
        struct MetropolisTransport
        {
            bool visualizeWeights    = false;
            bool visualizeStrategies = false;
        };
        struct SimplePath
        {
            bool sampleBSDF   = true;
            bool sampleLights = true;
        };
        struct StocProgPhotMap
        {
            int32_t  photonsPerIteration = -1;
            float    radius              = 0;
            uint32_t seed                = 0;
        };
        union Params
        {
            AmbientOcclusion    ao;
            BiDirPathTracing    bdpt;
            MetropolisTransport mlt;
            SimplePath          simplePath;
            StocProgPhotMap     sppm;
        };
        Params params;

        uint32_t maxDepth = 5; // all but ambient occlusion

        ELightSampler   lightSampler = ELightSampler::eBVH; // path, volpath, gpu, wavefront
        EIntegratorType type         = EIntegratorType::eVolPath;
    };

    // Acceletators ---------------------------------------------------------------------------------------------------
    enum class EAcceletatorType : uint8_t
    {
        eBVH = 0,
        eKdTree,
        eCount
    };

    EAcceletatorType acceleratorTypeFromStr(char const* str);

    enum class EBVHSplitMethod : uint8_t
    {
        eSAH = 0,
        eMiddle,
        eEqual,
        eHLBVH,
        eCount
    };

    EBVHSplitMethod bvhSplitMethodFromStr(char const* str);

    struct AcceleratorSpec
    {
        struct BVH
        {
            uint32_t        maxNodePrims = 4;
            EBVHSplitMethod splitMethod  = EBVHSplitMethod::eSAH;
        };
        struct KDTree
        {
            uint32_t intersectCost = 5;
            uint32_t traversalCost = 1;
            float    emptyBonus    = 0.5f;
            uint32_t maxPrims      = 1;
            int32_t  maxDepth      = -1;
        };
        union Params
        {
            BVH    bvh;
            KDTree kdtree;
        };

        Params           params;
        EAcceletatorType type;
    };

    // WorldBegin -----------------------------------------------------------------------------------------------------
    // Participating Media -------------------------------------------------------------
    // present both in global fragment (where the camera ray starts in) and after the `WorldBegin`
    // Parsing --------------------------------------------------------------------------------------------------------
    class WordParser
    {
    public:
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
        char     m_buffer[256]{};
        char     m_escapedBuffer[64]{};
        bool     m_needsContinuation = false;
        bool     m_haveEscaped       = false;
    };

    enum class EHeaderTokenType : uint8_t
    {
        eCamera,
        eSampler,
        eColorSpace,
        eFilm,
        eFilter,
        eIntegrator,
        eAccelerator,
        // eNamedMedia, TODO
        eCount
    };

    inline constexpr std::strong_ordering operator<=>(EHeaderTokenType a, EHeaderTokenType b)
    {
        return toUnderlying(a) <=> toUnderlying(b);
    }

    class HeaderTokenizer
    {
    private:
        static constexpr uint32_t size = std::max(
            {sizeof(CameraSpec),
             sizeof(SamplerSpec),
             sizeof(ColorSpaceSpec),
             sizeof(FilmSpec),
             sizeof(FilterSpec),
             sizeof(IntegratorSpec),
             sizeof(AcceleratorSpec)});
        static constexpr uint32_t alignment = std::max(
            {alignof(CameraSpec),
             alignof(SamplerSpec),
             alignof(ColorSpaceSpec),
             alignof(FilmSpec),
             alignof(FilterSpec),
             alignof(IntegratorSpec),
             alignof(AcceleratorSpec)});

    public:
        struct Storage
        {
            alignas(alignment) std::array<unsigned char, size> bytes;
        };

        HeaderTokenizer(std::string_view prevChunk, uint32_t prevOffset, std::string_view currChunk) :
        m_prevChunk(prevChunk),
        m_currChunk(currChunk),
        m_prevOffset(prevOffset)
        {
        }

        void    advance();
        bool    hasToken() const;
        Storage retrieveToken(EHeaderTokenType& outTokenType) const;
        size_t  offsetFromCurrent() const;

    private:
        bool parseNext(std::string_view* pChunk, size_t& inOutoffset);

        Storage          m_storage;
        std::string_view m_prevChunk;
        std::string_view m_currChunk;
        size_t           m_prevOffset;
        size_t           m_offset       = 0ULL; // relative to prevOffset
        EHeaderTokenType m_currentToken = EHeaderTokenType::eCount;
        uint32_t         m_started  : 1 = false;
        uint32_t         m_useCurr  : 1 = false;
        uint32_t         m_finished : 1 = false;
    };
}

DMT_MODULE_EXPORT dmt::model {}

DMT_MODULE_EXPORT dmt::job {
    struct ParseSceneHeaderData
    {
        std::string_view      filePath;
        dmt::AppContext*      actx;
        Options*              pInOutOptions; // when job kicked, you caller must wait on atomic
        std::atomic<uint32_t> done;          // should be zero when the job is kicked
        uint32_t              numChunkWorldBegin;
        uint32_t              offsetWorldBegin;
        uint32_t              numChunks;
    };

    void parseSceneHeader(uintptr_t address);
}