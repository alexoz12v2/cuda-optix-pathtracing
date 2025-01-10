#pragma once

#include "dmtmacros.h"

#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp>  // glm::pi
#include <glm/mat4x4.hpp>                // glm::mat4
#include <glm/vec3.hpp>                  // glm::vec3
#include <glm/vec4.hpp>                  // glm::vec4

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

#define DMT_INTERFACE_AS_HEADER
#include <platform/platform.h>

// TODO switch all structures with stack allocator
// TODO remove default values from clases and leave them only in parsing functions

// stuff related to .pbrt file parsing + data structures
DMT_MODULE_EXPORT dmt {
    // TODO move somewhere else
    class Transform
    {
    public:
        glm::mat4 m;    // Transformation matrix
        glm::mat4 mInv; // Inverse transformation matrix

        // Default constructor
        Transform() : m(glm::mat4(1.0f)), mInv(glm::mat4(1.0f)) {}

        // Constructor with an initial matrix
        explicit Transform(glm::mat4 const& matrix) : m(matrix), mInv(glm::inverse(matrix)) {}

        // Apply translation
        void translate_(glm::vec3 const& translation)
        {
            m    = glm::translate(m, translation);
            mInv = glm::translate(mInv, -translation);
        }

        // Apply scaling
        void scale_(glm::vec3 const& scaling)
        {
            m    = glm::scale(m, scaling);
            mInv = glm::scale(mInv, 1.0f / scaling);
        }

        // Apply rotation (angle in degrees)
        void rotate_(float angle, glm::vec3 const& axis)
        {
            m    = glm::rotate(m, glm::radians(angle), axis);
            mInv = glm::rotate(mInv, -glm::radians(angle), axis);
        }

        // Combine with another transform
        Transform combine(Transform const& other) const
        {
            Transform result;
            result.m    = m * other.m;
            result.mInv = other.mInv * mInv;
            return result;
        }

        // Combine with another transform
        void combine_(Transform const& other)
        {
            m    = m * other.m;
            mInv = other.mInv * mInv;
        }

        // Reset to identity matrix
        void reset()
        {
            m    = glm::mat4(1.0f);
            mInv = glm::mat4(1.0f);
        }

        // Swap m and mInv
        void inverse() { std::swap(m, mInv); }

        // Apply the transform to a point
        glm::vec3 applyToPoint(glm::vec3 const& point) const
        {
            glm::vec4 result = m * glm::vec4(point, 1.0f);
            return glm::vec3(result);
        }

        // Apply the inverse transform to a point
        glm::vec3 applyInverseToPoint(glm::vec3 const& point) const
        {
            glm::vec4 result = mInv * glm::vec4(point, 1.0f);
            return glm::vec3(result);
        }

        // Equality comparison
        bool operator==(Transform const& other) const { return m == other.m && mInv == other.mInv; }

        // Inequality comparison
        bool operator!=(Transform const& other) const { return !(*this == other); }
    };

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
    enum class ECameraType : uint8_t
    {
        eOrthographic = 0,
        ePerspective,
        eRealistic,
        eSpherical,
        eCount
    };

    enum class ESphericalMapping : uint8_t
    {
        eEqualArea = 0,
        eEquirectangular,
        eCount
    };

    namespace apertures {
        using namespace std::string_view_literals;
        static inline constexpr std::string_view gaussian = "gaussian"sv;
        static inline constexpr std::string_view square   = "square"sv;
        static inline constexpr std::string_view pentagon = "pentagon"sv;
        static inline constexpr std::string_view star     = "star"sv;
    }

    struct CameraSpec
    {
    public:
        static constexpr float invalidScreen      = -std::numeric_limits<float>::infinity();
        static constexpr float invalidAspectRatio = -1.f;
        CameraSpec()                              = default;
        // Since realistic camrea stores two filenames as strings, we cannot use memcpy for copy semantics
        CameraSpec(CameraSpec const&);
        CameraSpec(CameraSpec&&) noexcept;
        CameraSpec& operator=(CameraSpec const&);
        CameraSpec& operator=(CameraSpec&&) noexcept;
        ~CameraSpec() noexcept
        {
            if (type == ECameraType::eRealistic)
            {
                std::destroy_at(&params.r.lensfile);
                std::destroy_at(&params.r.aperture);
            }
        }

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

    enum class ERandomization : uint8_t
    {
        eFastOwen = 0,
        eNone,
        ePermuteDigits,
        eOwen,
        eCount
    };

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

    enum class EGVufferCoordSys : uint8_t
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
    enum class EFilterType : uint8_t
    {
        eGaussian = 0,
        eBox,
        eMitchell,
        eSinc,
        eTriangle,
        eCount
    };

    float defaultRadiusFromFilterType(EFilterType e);
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

    enum class ELightSampler : uint8_t
    {
        eBVH = 0,
        eUniform,
        ePower,
        eCount
    };

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

    enum class EBVHSplitMethod : uint8_t
    {
        eSAH = 0,
        eMiddle,
        eEqual,
        eHLBVH,
        eCount
    };

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

    // ----------------------------------------------------------------------------------------------------------------
    using ArgsDArray = std::vector<std::string>;
    struct ParamPair
    {
        using ValueList = ArgsDArray;
        constexpr explicit ParamPair(sid_t type) : type(type) {}

        void addParamValue(std::string_view value) { values.emplace_back(value); }

        std::string_view valueAt(uint32_t i) const { return values[i]; }

        uint32_t numParams() const { return static_cast<uint32_t>(values.size()); }

        // TODO better (char buffer fixed)
        ArgsDArray values;

        sid_t type;
    };

    using ParamMap = std::map<sid_t, ParamPair>;

    enum class ETarget : uint8_t
    {
        eShape = 0,
        eLight,
        eMaterial,
        eMedium,
        eTexture,
        eCount
    };

    enum class ETextureType : uint8_t
    {
        eSpectrum = 0,
        eFloat
    };

    enum class ETextureClass : uint8_t
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

    enum class EMaterialType : uint8_t
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

    enum class ELightType : uint8_t
    {
        eDistant = 0,
        eGoniometric,
        eInfinite,
        ePoint,
        eProjection,
        eSpot,
        eCount
    };

    enum class EAreaLightType : uint8_t
    {
        eDiffuse = 0,
        eCount
    };

    enum class EShapeType : uint8_t
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

    enum class EActiveTransform : uint8_t
    {
        eStartTime = 0,
        eEndTime,
        eAll,
        eCount
    };

    // LightSource ----------------------------------------------------------------------------------------------------
    struct LightSourceSpec
    {
        // 4 byte aligned, coommon
        union PowerOrIlluminance
        { // there is no default, one of these must be present
            // all except distant and infinite
            float power;
            // distant, infinite
            float illuminance;
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

        TokenStream(AppContext& actx, std::string_view filePath);
        TokenStream(TokenStream const&)                = delete;
        TokenStream(TokenStream&&) noexcept            = delete;
        TokenStream& operator=(TokenStream const&)     = delete;
        TokenStream& operator=(TokenStream&&) noexcept = delete;
        ~TokenStream() noexcept;

        std::string next(AppContext& actx);
        void        advance(AppContext& actx);
        std::string peek();

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

    struct EndOfHeaderInfo
    {
        CameraSpec cameraSpec;
    };

    class DMT_INTERFACE IParserTarget
    {
    public:
        virtual void Scale(float sx, float sy, float sz) = 0;

        virtual void Shape(EShapeType type, ParamMap const& params) = 0;

        virtual ~IParserTarget() {};

        virtual void Option(sid_t name, ParamPair const& value) = 0;

        virtual void Identity()                                        = 0;
        virtual void Translate(float dx, float dy, float dz)           = 0;
        virtual void Rotate(float angle, float ax, float ay, float az) = 0;
        virtual void LookAt(float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz) = 0;
        virtual void ConcatTransform(std::array<float, 16> const& transform) = 0;
        virtual void Transform(std::array<float, 16> transform)              = 0;
        virtual void CoordinateSystem(sid_t name)                            = 0;
        virtual void CoordSysTransform(sid_t name)                           = 0;
        virtual void ActiveTransformAll()                                    = 0;
        virtual void ActiveTransformEndTime()                                = 0;
        virtual void ActiveTransformStartTime()                              = 0;
        virtual void TransformTimes(float start, float end)                  = 0;

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
    struct TransformSet
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
        float        transfromStartTime  = 0.f;
        float        transformEndTime    = 1.f;

        EColorSpaceType colorSpace = EColorSpaceType::eSRGB;
    };

    class SceneDescription : public IParserTarget
    {
    public:
        void Scale(float sx, float sy, float sz) override;
        void Shape(EShapeType type, ParamMap const& params) override;
        void Option(sid_t name, ParamPair const& value) override;
        void Identity() override;
        void Translate(float dx, float dy, float dz) override;
        void Rotate(float angle, float ax, float ay, float az) override;
        void LookAt(float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz) override;
        void ConcatTransform(std::array<float, 16> const& transform) override;
        void Transform(std::array<float, 16> transform) override;
        void CoordinateSystem(sid_t name) override;
        void CoordSysTransform(sid_t name) override;
        void ActiveTransformAll() override;
        void ActiveTransformEndTime() override;
        void ActiveTransformStartTime() override;
        void TransformTimes(float start, float end) override;
        void ColorSpace(EColorSpaceType colorSpace) override;
        void PixelFilter(FilterSpec const& spec) override;
        void Film(FilmSpec const& spec) override;
        void Accelerator(AcceleratorSpec const& spec) override;
        void Integrator(IntegratorSpec const& spec) override;
        void Camera(CameraSpec const& params) override;
        void MakeNamedMedium(sid_t name, ParamMap const& params) override;
        void MediumInterface(sid_t insideName, sid_t outsideName) override;
        void Sampler(SamplerSpec const& spec) override;
        void WorldBegin() override;
        void AttributeBegin() override;
        void AttributeEnd() override;
        void Attribute(ETarget target, ParamMap const& params) override;
        void Texture(sid_t name, ETextureType type, ETextureClass texname, ParamMap const& params) override;
        void Material(EMaterialType type, ParamMap const& params) override;
        void MakeNamedMaterial(sid_t name, ParamMap const& params) override;
        void NamedMaterial(sid_t name) override;
        void LightSource(ELightType type, ParamMap const& params) override;
        void AreaLightSource(EAreaLightType type, ParamMap const& params) override;
        void ReverseOrientation() override;
        void ObjectBegin(sid_t name) override;
        void ObjectEnd() override;
        void ObjectInstance(sid_t name) override;
        void EndOfOptions(Options const& options) override;
        void EndOfFiles() override;
        void EndOfHeader(EndOfHeaderInfo const& info) override;

    public:
        CameraSpec      cameraSpec;
        SamplerSpec     samplerSpec;
        ColorSpaceSpec  colorSpaceSpec;
        FilmSpec        filmSpec;
        FilterSpec      filterSpec;
        IntegratorSpec  integratorSpec;
        AcceleratorSpec acceleratorSpec;

    private:
        GraphicsState graphicsState;
    };

    enum class EParsingStep : uint8_t
    {
        eOptions = 0,
        eHeader,
        eWorld
    };

    enum class EScope : uint8_t
    {
        eAttribute = 0,
        eObject
    };

    enum class EEncounteredHeaderDirective : uint32_t
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
        return toUnderlying(e) & toUnderlying(val) != 0;
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
        SceneParser(AppContext& actx, IParserTarget* pTarget, std::string_view filePath);
        void parse(AppContext& actx, Options& inOutOptions);

    private:
        static bool setOptionParam(AppContext& actx, ParamMap const& params, Options& outOptions);

    private:
        uint32_t     parseArgs(AppContext& actx, TokenStream& stream, ArgsDArray& outArr);
        uint32_t     parseParams(AppContext& actx, TokenStream& stream, ParamMap& outParams);
        bool         transitionToHeaderIfFirstHeaderDirective(AppContext&                 actx,
                                                              Options const&              outOptions,
                                                              EEncounteredHeaderDirective val);
        void         pushFile(AppContext& actx, std::string_view filePath, bool isImportOrMainFile);
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
}

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