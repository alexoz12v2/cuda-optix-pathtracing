#pragma once

#include "dmtmacros.h"

#include <array>
#include <atomic>
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

#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi

#if defined(DMT_INTERFACE_AS_HEADER)
#include <platform/platform.h>
#else
import platform;
#endif

// TODO switch all structures with stack allocator
// TODO remove default values from clases and leave them only in parsing functions

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

    class OneShotStackMemoryResource : public std::pmr::memory_resource
    {
    public:
        OneShotStackMemoryResource(MemoryContext* memoryContext, size_t alignment = alignof(std::max_align_t)) :
        m_memoryContext(memoryContext),
        m_alignment(alignment),
        m_allocated(false)
        {
            assert(m_memoryContext);
        }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override
        {
            if (m_allocated)
            { // Subsequent allocations are not allowed.
                return nullptr;
            }

            // Use the stack allocator for the first allocation.
            void* ptr = m_memoryContext->stackAllocate(bytes, std::max(alignment, m_alignment), EMemoryTag::eUnknown, 0);
            m_allocated = true;
            return ptr;
        }

        void do_deallocate(void* /*ptr*/, size_t /*bytes*/, size_t /*alignment*/) override
        {
            // Deallocate does nothing.
        }

        bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
        {
            // Equality based on type and MemoryContext pointer.
            auto otherResource = dynamic_cast<OneShotStackMemoryResource const*>(&other);
            return otherResource && otherResource->m_memoryContext == m_memoryContext;
        }

    private:
        /**
         * Exception to the rule dependency injection through function parameter passing, as required
         * by the `std::memory_resource` interface
         */
        MemoryContext* m_memoryContext; // Pointer to the MemoryContext used for allocations.
        size_t         m_alignment;     // Alignment for stack allocation.
        bool           m_allocated;     // Whether the first allocation has already occurred.
    };

    template <typename T>
    class StackArrayDeleter
    {
    public:
        void operator()(T* ptr) const {}
    };

    // ----------------------------------------------------------------------------------------------------------------

    struct Bitmap
    {
        Bitmap() : map(0) {}

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

        bool setValue(uint64_t value, uint64_t index) { return cas(value, index, false, 0, 0); }

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
        bool remove(MemoryContext& mctx, uint32_t key);

    private:
        void cleanupINode(MemoryContext& mctx, INode* inode, void (*dctor)(MemoryContext& mctx, void* value));
        void cleanupSNode(MemoryContext& mctx, SNode* inode, void (*dctor)(MemoryContext& mctx, void* value));
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

    inline bool wavefrontOrGPU(Options const& options)
    {
        return hasFlag(options.flags, EBoolOptions::ecUseGPU | EBoolOptions::ecWavefront);
    }

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
        CameraSpec()                                 = default;
        CameraSpec(CameraSpec const&)                = default;
        CameraSpec(CameraSpec&&) noexcept            = default;
        CameraSpec& operator=(CameraSpec const&)     = default;
        CameraSpec& operator=(CameraSpec&&) noexcept = default;
        ~CameraSpec() noexcept
        {
            if (type == ECameraType::eRealistic)
            {
                std::destroy_at(&params.r.lensfile);
                std::destroy_at(&params.r.aperture);
            }
        }

        struct Projecting           // perspective or orthographic
        {                           // params are all lowercase in the file
            float frameAspectRatio; // computed from film
            // [-1, 1] along shorter axis, [-screnWindow, +screenWindow] in longer axis. Default = aspect ratio if > 1, otherwise 1/aspect ratio
            float screenWindow;
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

        // 8 Byte aligned
        Params params;

        // 4 byte aligned
        float shutteropen  = 0;
        float shutterclose = 1;

        // 1 byte aligned
        ECameraType type = ECameraType::ePerspective;
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
            int32_t x      = 4;
            int32_t y      = 4;
            bool    jitter = true;
        };

        // everyone
        int32_t seed; // default from options, file takes precedence

        union Samples
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
        EColorSpaceType type = EColorSpaceType::eSRGB;
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
    enum class EFilterType : uint8_t
    {
        eGaussian = 0,
        eBox,
        eMitchell,
        eSinc,
        eTriangle,
        eCount
    };

    EFilterType filterTypeFromStr(char const* str);
    float       defaultRadiusFromFilterType(EFilterType e);

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
        struct MetropolisTransport
        {
            float   sigma                  = 0.01f;
            float   largestStepProbability = 0.3f;
            int32_t mutationsPerPixel      = 100;
            int32_t chains                 = 1000;
            int32_t bootstraqpSamples      = 100000;
        };
        struct BiDirPathTracing
        {
            bool visualizeStrategies = false;
            bool visualizeWeights    = false;
        };
        struct SimplePath
        {
            bool sampleBSDF   = true;
            bool sampleLights = true;
        };
        struct StocProgPhotMap
        {
            int32_t photonsPerIteration = -1;
            float   radius              = 0;
            int32_t seed                = 0;
        };
        union Params
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
            int32_t         maxNodePrims = 4;
            EBVHSplitMethod splitMethod  = EBVHSplitMethod::eSAH;
        };
        struct KDTree
        {
            int32_t intersectCost = 5;
            int32_t traversalCost = 1;
            float   emptyBonus    = 0.5f;
            int32_t maxPrims      = 1;
            int32_t maxDepth      = -1;
        };
        union Params
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
    class WordParser
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

    struct ArrayData
    {
        static constexpr uint32_t countPenNode = 248u / sizeof(int32_t);
        union U
        {
            std::array<int32_t, countPenNode> is;
            std::array<float, countPenNode>   fs;
        };
        U arr;
    };
    template struct PoolNode<ArrayData, EBlockSize::e256B>;
    using ArrayNode256B = PoolNode<ArrayData, EBlockSize::e256B>;

    // ----------------------------------------------------------------------------------------------------------------
    enum class ETypeModifier : uint8_t
    {
        eEmpty          = 0,
        eScalar         = 1u << 0u,
        eArray          = 1u << 1u,
        eUnboundedArray = 0u << 2u,
        e1Elem          = 1u << 2u,
        e2Elem          = 2u << 2u,
        e3Elem          = 3u << 2u,
        eTexture        = 1u << 4u,
        eEnum           = 1u << 5u,
    };
    constexpr ETypeModifier operator|(ETypeModifier lhs, ETypeModifier rhs) noexcept
    {
        return static_cast<ETypeModifier>(toUnderlying(lhs) | toUnderlying(rhs));
    }
    constexpr ETypeModifier operator&(ETypeModifier lhs, ETypeModifier rhs) noexcept
    {
        return static_cast<ETypeModifier>(toUnderlying(lhs) & toUnderlying(rhs));
    }
    constexpr ETypeModifier operator~(ETypeModifier value) noexcept
    {
        return static_cast<ETypeModifier>(~toUnderlying(value));
    }
    constexpr ETypeModifier operator^(ETypeModifier lhs, ETypeModifier rhs) noexcept
    {
        return static_cast<ETypeModifier>(toUnderlying(lhs) ^ toUnderlying(rhs));
    }
    constexpr ETypeModifier& operator|=(ETypeModifier& lhs, ETypeModifier rhs) noexcept
    {
        lhs = lhs | rhs;
        return lhs;
    }
    constexpr ETypeModifier& operator&=(ETypeModifier& lhs, ETypeModifier rhs) noexcept
    {
        lhs = lhs & rhs;
        return lhs;
    }
    constexpr ETypeModifier& operator^=(ETypeModifier& lhs, ETypeModifier rhs) noexcept
    {
        lhs = lhs ^ rhs;
        return lhs;
    }

    struct TypeTuple
    {
        static constexpr uint32_t              maxNumTypes = 3;
        std::array<sid_t, maxNumTypes>         sids;
        uint64_t                               count;
        std::array<ETypeModifier, maxNumTypes> mods;
        unsigned char                          padding[5];
    };
    static_assert(sizeof(TypeTuple) == 40 && alignof(TypeTuple) == 8);

    struct Parameter
    {
        Parameter(TypeTuple tup) : allowedTypes(tup) {}
        TypeTuple allowedTypes;
        // sid_t     sid; implicitly stored as map key
    };

    struct Argument
    {
        sid_t type;
    };

    enum class EDirectivePos : uint8_t
    {
        eNothing        = 0,
        eHeaderBlock    = 1u << 0u,
        eWorldBlock     = 1u << 1u,
        eAttributeBlock = 1u << 2u, // note: attributes can be nested
        eObjectBlock    = 1u << 3u,
        eAll            = std::numeric_limits<uint8_t>::max()
    };
    constexpr EDirectivePos operator|(EDirectivePos lhs, EDirectivePos rhs) noexcept
    {
        return static_cast<EDirectivePos>(toUnderlying(lhs) | toUnderlying(rhs));
    }
    constexpr EDirectivePos operator&(EDirectivePos lhs, EDirectivePos rhs) noexcept
    {
        return static_cast<EDirectivePos>(toUnderlying(lhs) & toUnderlying(rhs));
    }
    constexpr EDirectivePos operator~(EDirectivePos value) noexcept
    {
        return static_cast<EDirectivePos>(~toUnderlying(value));
    }
    constexpr EDirectivePos operator^(EDirectivePos lhs, EDirectivePos rhs) noexcept
    {
        return static_cast<EDirectivePos>(toUnderlying(lhs) ^ toUnderlying(rhs));
    }
    constexpr EDirectivePos& operator|=(EDirectivePos& lhs, EDirectivePos rhs) noexcept
    {
        lhs = lhs | rhs;
        return lhs;
    }
    constexpr EDirectivePos& operator&=(EDirectivePos& lhs, EDirectivePos rhs) noexcept
    {
        lhs = lhs & rhs;
        return lhs;
    }
    constexpr EDirectivePos& operator^=(EDirectivePos& lhs, EDirectivePos rhs) noexcept
    {
        lhs = lhs ^ rhs;
        return lhs;
    }
    constexpr bool canBeInHeaderBlock(EDirectivePos pos) noexcept
    {
        return (pos & EDirectivePos::eHeaderBlock) == EDirectivePos::eHeaderBlock;
    }
    constexpr bool canBeInWorldBlock(EDirectivePos pos) noexcept
    {
        return (pos & EDirectivePos::eWorldBlock) == EDirectivePos::eWorldBlock;
    }

    struct Directive
    {
        using ParamsMap = std::pmr::map<sid_t, Parameter>;
        Directive(std::pmr::memory_resource* mem) :
        args(mem),
        allowedParams(mem),
        maxParams(0),
        allowedPos(EDirectivePos::eNothing)
        {
        }
        std::pmr::vector<Argument> args;
        ParamsMap                  allowedParams;
        uint32_t                   maxParams; // 0 means unlimited, and if there are duplicates consider the last one
        EDirectivePos              allowedPos;
    };

    enum class EHeaderBlockState : uint32_t
    {
        eNone = 0,
        eDirectiveRead,
        eArgsReading,
        eArgsRead,
        eParamsReading,
        eInsideArray,
    };

    struct ParamPair
    {
        using ValueList = std::vector<std::string>;
        constexpr explicit ParamPair(sid_t type) : type(type) {}

        void addParamValue(std::string_view value) { values.emplace_back(value); }

        std::string_view valueAt(uint32_t i) const { return values[i]; }

        uint32_t numParams() const { return static_cast<uint32_t>(values.size()); }

        // TODO better (char buffer fixed)
        std::vector<std::string> values;

        sid_t type;
    };

    using ParamMap = std::map<sid_t, ParamPair>;
    enum class EScope
    {
        eAttribute,
        eObject,
    };

    /**
     * surely pool allocated
     */
    class HeaderTokenizer
    {
    public:
        HeaderTokenizer(Options const& cmdOptions) : fileOptions(cmdOptions) {}

        struct Ret
        {
            size_t   worldBlockOffset;
            uint32_t numWorldBlockChunk;
        };
        Ret parseHeader(AppContext& actx, std::string_view filePath);

        Options         fileOptions;
        CameraSpec      cameraSpec;
        SamplerSpec     samplerSpec;
        ColorSpaceSpec  colorSpaceSpec;
        FilmSpec        filmSpec;
        FilterSpec      filterSpec;
        IntegratorSpec  integratorSpec;
        AcceleratorSpec acceleratorSpec;
        // TODO list of media/materials/any declaration the header can have

    private:
        void parseFile(AppContext& actx, std::string_view basePath, std::string_view includeArgument);
        void consumeToken(AppContext& actx, std::string_view basePath, std::string_view token);
        void reestState();

        void setCurrentParam(sid_t type, sid_t name);
        void setClassArg(std::string_view token, sid_t arg);
        void handleDirective(AppContext& actx, sid_t directive);
        void handleArgument(AppContext& actx, std::string_view token);
        void handleFirstTokenAfterDirective(AppContext& actx, std::string_view token);
        void insertParameterValue(AppContext& actx, sid_t name, std::string_view token);

        // TODO maybe
        std::map<sid_t, ParamPair>              m_params;
        std::stack<EScope, std::vector<EScope>> m_blockStack;
        EHeaderBlockState                       m_state = EHeaderBlockState::eNone;
        std::string_view                        m_dequotedArg;
        sid_t                                   m_insideDirective  = 0;
        sid_t                                   m_currentParamType = 0;
        sid_t                                   m_currentParamName = 0;
        sid_t                                   m_classArg         = 0;
        uint32_t                                m_argsRead         = 0;
        uint32_t                                m_paramsRead       = 0;
        bool                                    m_inWorldBlock     = false;
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