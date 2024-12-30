module;

#include "dmtmacros.h"

#include <array>
#include <atomic>
#include <bit>
#include <string_view>
#include <type_traits>

#include <cassert>
#include <cctype>
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
        static constexpr std::array<std::string_view, count> types{"fastowen"sv, "none"sv, "permutedigits"sv, "owen"sv};

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

    EFilterType filterTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EFilterType> count = toUnderlying(EFilterType::eCount);
        static constexpr std::array<std::string_view, count> types{
            "box"sv,
            "gaussian"sv,
            "mitchell"sv,
            "sinc"sv,
            "triangle"sv,
        };

        return ::enumFromStr(str, types, EFilterType::eGaussian);
    }

    float defaultRadiusFromFilterType(EFilterType e)
    {
        switch (e)
        {
            using enum EFilterType;
            case eBox:
                return 0.5f;
            case eMitchell:
                return 2.f;
            case eSinc:
                return 4.f;
            case eTriangle:
                return 2.f;
            case eGaussian:
                [[fallthrough]];
            default:
                return 1.5f;
        }
    }

    EIntegratorType integratorTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EIntegratorType> count = toUnderlying(EIntegratorType::eCount);
        static constexpr std::array<std::string_view, count>
            types{"volpath"sv,
                  "ambientocclusion"sv,
                  "bdpt"sv,
                  "lightpath"sv,
                  "mlt"sv,
                  "path"sv,
                  "randomwalk"sv,
                  "simplepath"sv,
                  "simplevolpath"sv,
                  "sppm"sv};

        return ::enumFromStr(str, types, EIntegratorType::eVolPath);
    }

    ELightSampler lightSamplerFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ELightSampler> count = toUnderlying(ELightSampler::eCount);
        static constexpr std::array<std::string_view, count>   types{"bvh"sv, "uniform"sv, "power"sv};

        return ::enumFromStr(str, types, ELightSampler::eBVH);
    }

    EAcceletatorType acceleratorTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EAcceletatorType> count = toUnderlying(EAcceletatorType::eCount);
        static constexpr std::array<std::string_view, count>      types{"bvh"sv, "kdtree"sv};

        return ::enumFromStr(str, types, EAcceletatorType::eBVH);
    }

    EBVHSplitMethod bvhSplitMethodFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EBVHSplitMethod> count = toUnderlying(EBVHSplitMethod::eCount);
        static constexpr std::array<std::string_view, count>     types{"sah"sv, "middle"sv, "equal"sv, "hlbvh"sv};

        return ::enumFromStr(str, types, EBVHSplitMethod::eSAH);
    }

    // CTrie ----------------------------------------------------------------------------------------------------------
    static constexpr uintptr_t getAtomicCounterAddress(uintptr_t elemAddr, size_t elementSize, size_t paddingEnd)
    {
        return elemAddr + elementSize - sizeof(std::atomic<uint32_t>) - paddingEnd;
    }

    static constexpr uintptr_t getKeyAddress(uintptr_t elemAddr, size_t elementSize, size_t paddingEnd)
    {
        return elemAddr + elementSize - sizeof(uint32_t) - sizeof(std::atomic<uint32_t>) - paddingEnd;
    }

    CTrie::CTrie(MemoryContext& mctx, AllocatorTable const& table, size_t valueSize, size_t valueAlign) :
    m_table(table),
    m_root(taggedNullptr),
    m_size(0),
    m_sNodeSize(0),
    m_sNodeAlign(std::max(valueAlign, alignof(std::atomic<uint32_t>)))
    {
        m_paddingAfterValue = (alignof(uint32_t) - (valueSize % alignof(uint32_t))) % alignof(uint32_t);
        m_paddingEnd        = (valueSize + sizeof(uint32_t) + sizeof(std::atomic<uint32_t>)) % m_sNodeAlign;
        m_sNodeSize = valueSize + m_paddingAfterValue + sizeof(uint32_t) + sizeof(std::atomic<uint32_t>) + m_paddingEnd;

        m_root = m_table.allocate(mctx, sizeof(SNode), alignof(SNode));
        if (m_root == taggedNullptr)
        {
            mctx.pctx.error("Couldn't allocate first block for CTrie");
            mctx.pctx.dbgErrorStackTrace();
            std::abort();
        }
        INode* pRoot = std::bit_cast<INode*>(m_table.rawPtr(m_root));
        std::memset(pRoot, 0, sizeof(INode));
        pRoot->bitmap.store(std::numeric_limits<uint64_t>::max(), std::memory_order_release);
    }

    void CTrie::cleanup(MemoryContext& mctx, void (*dctor)(MemoryContext& mctx, void* ptr))
    {
        // Recursive lambda for cleaning up nodes
        auto cleanupNode = [&](auto&& self, TaggedPointer taggedPtr, bool _isINode) -> void
        {
            if (taggedPtr == taggedNullptr) // Skip null nodes
                return;

            void* rawPtr = m_table.rawPtr(taggedPtr);

            if (_isINode)
            {
                INode* iNode = std::bit_cast<INode*>(rawPtr);

                // Traverse all children of the INode
                for (uint32_t i = 0; i < 32; ++i)
                {
                    TaggedPointer childPtr = iNode->children[i];
                    // Determine if the child is an INode or SNode
                    bool childIsINode = isINode(iNode, i);
                    self(self, childPtr, childIsINode);
                }

                // Free the INode
                m_table.free(mctx, taggedPtr, sizeof(INode), alignof(INode));
            }
            else
            {
                SNode* sNode = std::bit_cast<SNode*>(rawPtr);

                // Call destructor on all elements in the SNode
                size_t maxIdx = computeMaxElements(256, m_sNodeSize, m_sNodeAlign);
                for (uint32_t i = 0; i < maxIdx; ++i)
                {
                    if (!isFree(sNode, i)) // Only call destructor for non-free slots (TODO lock for write)
                    {
                        lockForWrite(sNode, i);
                        void* pData    = sNode->data;
                        void* dataAddr = std::bit_cast<void*>(std::bit_cast<uintptr_t>(pData) + i * m_sNodeSize);
                        dctor(mctx, dataAddr); // Call the user-provided destructor
                    }
                    else
                    {
                        lockForWrite(sNode, i);
                    }
                }

                // Free the SNode
                m_table.free(mctx, taggedPtr, sizeof(SNode), alignof(SNode));
            }
        };

        // Start cleanup from the root
        cleanupNode(cleanupNode, m_root, true);

        // Reset the root pointer
        m_root = taggedNullptr;
    }

    bool CTrie::insert(MemoryContext& mctx, uint32_t keyHash, void const* pValue)
    {
        INode*  pRoot = std::bit_cast<INode*>(m_table.rawPtr(m_root));
        EResult res   = EResult::eRestart;
        while (res == EResult::eRestart)
        {
            res = iinsert(mctx, pRoot, keyHash, pValue);
        }
        return res != EResult::eError;
    }

    void const* CTrie::lookupConstRef(uint32_t keyHash)
    {
        // the given 32 bit index is split into packets of 5 bits
        uint32_t mask   = 0xF800'0000;
        INode*   parent = std::bit_cast<INode*>(m_table.rawPtr(m_root));
        return lookupConstRefFrom(parent, mask, keyHash);
    }

    void CTrie::finishRead(void const** ppElem)
    {
        if (!ppElem || !*ppElem)
            return;

        uintptr_t elemAddr = std::bit_cast<uintptr_t>(*ppElem);

        // Calculate the SNode and index from the element address
        uintptr_t   sNodeAddr = elemAddr & ~(m_sNodeSize - 1);
        SNode*      sNode     = std::bit_cast<SNode*>(sNodeAddr);
        void const* pData     = sNode->data;
        uint32_t    index     = (elemAddr - std::bit_cast<uintptr_t>(pData)) / m_sNodeSize;

        // Mark the slot as free (set the bits in the bitmap to 0b00)
        unlockForRead(sNode, index, m_sNodeSize, m_paddingEnd);

        // Reset the pointer
        *ppElem = nullptr;
    }

    void CTrie::finishWrite(void** ppElem)
    {
        if (!ppElem || !*ppElem)
            return;

        uintptr_t elemAddr = std::bit_cast<uintptr_t>(*ppElem);

        // Calculate the SNode and index from the element address
        uintptr_t sNodeAddr = elemAddr & ~(m_sNodeSize - 1);
        SNode*    sNode     = std::bit_cast<SNode*>(sNodeAddr);
        void*     pData     = sNode->data;
        uint32_t  index     = (elemAddr - std::bit_cast<uintptr_t>(pData)) / m_sNodeSize;

        // Mark the slot as free (set the bits in the bitmap to 0b00)
        unlockForWrite(sNode, index);

        // Reset the pointer
        *ppElem = nullptr;
    }

    void CTrie::lookupCopy(uint32_t keyHash, void** ppStorage)
    {
        assert(ppStorage && *ppStorage);
        void const* elem = lookupConstRef(keyHash); // Perform the read-locked lookup
        if (!elem)
        {
            *ppStorage = nullptr;
            return;
        }

        // Create a copy of the value
        uintptr_t elemAddr  = std::bit_cast<uintptr_t>(elem);
        uint64_t  valueSize = getValueSize();

        std::memcpy(*ppStorage, std::bit_cast<void*>(elemAddr), valueSize);

        // Finish the read lock
        finishRead(&elem);
    }

    void const* CTrie::lookupConstRefFrom(INode* inode, uint32_t mask, uint32_t keyHash)
    {
        uint32_t shamt  = countTrailingZeros(mask);
        uint32_t packet = (keyHash & mask) >> shamt;
        mask >>= 5;

        if (isINode(inode, packet))
        {
            inode = std::bit_cast<INode*>(m_table.rawPtr(inode->children[packet]));
            return lookupConstRefFrom(inode, mask, keyHash);
        }
        else
        {
            SNode* self   = std::bit_cast<SNode*>(m_table.rawPtr(inode->children[packet]));
            size_t maxIdx = computeMaxElements(256, m_sNodeSize, m_sNodeAlign);

            for (uint32_t i = 0; i < maxIdx; ++i)
            {
                if (!isFullElement(self, i))
                    continue;

                lockForRead(self, i, m_sNodeSize, m_paddingEnd);

                void const* data     = self->data;
                uintptr_t   elemAddr = std::bit_cast<uintptr_t>(data) + i * m_sNodeSize;

                uintptr_t       keyAddr = getKeyAddress(elemAddr, m_sNodeSize, m_paddingEnd);
                uint32_t const& key     = *std::bit_cast<uint32_t const*>(keyAddr);

                if (key == keyHash)
                {
                    assert(elemAddr == alignToAddr(elemAddr, m_sNodeAlign));
                    return std::bit_cast<void const*>(elemAddr);
                }

                unlockForRead(self, i, m_sNodeSize, m_paddingEnd);
            }

            return nullptr;
        }
    }

    void* CTrie::lookupRefFrom(INode* inode, uint32_t mask, uint32_t keyHash)
    {
        uint32_t shamt  = countTrailingZeros(mask);
        uint32_t packet = (keyHash & mask) >> shamt;
        mask >>= 5;

        if (isINode(inode, packet))
        {
            inode = std::bit_cast<INode*>(m_table.rawPtr(inode->children[packet]));
            return lookupRefFrom(inode, mask, keyHash);
        }
        else
        {
            SNode* self   = std::bit_cast<SNode*>(m_table.rawPtr(inode->children[packet]));
            size_t maxIdx = computeMaxElements(256, m_sNodeSize, m_sNodeAlign);

            for (uint32_t i = 0; i < maxIdx; ++i)
            {
                if (!isFullElement(self, i))
                    continue;

                lockForWrite(self, i);

                void const* data     = self->data;
                uintptr_t   elemAddr = std::bit_cast<uintptr_t>(data) + i * m_sNodeSize;

                uintptr_t       keyAddr = getKeyAddress(elemAddr, m_sNodeSize, m_paddingEnd);
                uint32_t const& key     = *std::bit_cast<uint32_t const*>(keyAddr);

                if (key == keyHash)
                {
                    assert(elemAddr == alignToAddr(elemAddr, m_sNodeAlign));
                    return std::bit_cast<void*>(elemAddr);
                }

                unlockForWrite(self, i);
            }

            return nullptr;
        }
    }

    void* CTrie::lookupRef(uint32_t keyHash)
    {
        // TODO similiar to lookupConstRef, but return a mutable thing
        // the given 32 bit index is split into packets of 5 bits
        uint32_t mask   = 0xF800'0000;
        INode*   parent = std::bit_cast<INode*>(m_table.rawPtr(m_root));
        return lookupRefFrom(parent, mask, keyHash);
    }

    bool CTrie::remove(MemoryContext& mctx, uint32_t keyHash)
    {
        // TODO
        return false;
    }


    size_t CTrie::getValueSize() const
    {
        uint64_t valueSz = m_sNodeSize - m_paddingAfterValue - m_paddingEnd - sizeof(uint32_t) -
                           sizeof(std::atomic<uint32_t>);
        return valueSz;
    }

    CTrie::EResult CTrie::iinsert(MemoryContext& mctx, INode* pNode, uint32_t keyHash, void const* pValue)
    {
        uint32_t mask   = 0xF800'0000; // Initial mask for top-level (e.g., first 5 bits)
        uint32_t shamt  = countTrailingZeros(mask);
        uint32_t packet = (keyHash & mask) >> shamt;

        while (true)
        {
            if (isINode(pNode, packet))
            {
                pNode = std::bit_cast<INode*>(m_table.rawPtr(pNode->children[packet]));
                mask >>= 5;
                shamt  = countTrailingZeros(mask);
                packet = (keyHash & mask) >> shamt;
            }
            else
            {
                SNode* sNode    = std::bit_cast<SNode*>(m_table.rawPtr(pNode->children[packet]));
                size_t maxIdx   = computeMaxElements(256, m_sNodeSize, m_sNodeAlign);
                bool   inserted = false;

                if (isChildNotAllocated(pNode, packet))
                {
                    if (lockForAlloc(pNode, packet))
                    {
                        pNode->children[packet] = m_table.allocate(mctx, sizeof(SNode), alignof(SNode));
                        sNode                   = std::bit_cast<SNode*>(m_table.rawPtr(pNode->children[packet]));
                        std::memset(sNode, 0, sizeof(SNode));
                        unlockForAlloc(pNode, packet, 0b00);
                    }
                }

                if (!sNode)
                {
                    mctx.pctx.error("Couldn't allocate memory for a CTrie node");
                    return EResult::eError;
                }

                while (!lockForAlloc(pNode, packet, true))
                {
                    std::this_thread::yield();
                }

                for (uint32_t i = 0; i < maxIdx; ++i)
                {
                    if (!isFree(sNode, i))
                        continue;

                    lockForWrite(sNode, i);

                    void*     data     = sNode->data;
                    uintptr_t elemAddr = std::bit_cast<uintptr_t>(data) + i * m_sNodeSize;

                    uintptr_t keyAddr = getKeyAddress(elemAddr, m_sNodeSize, m_paddingEnd);
                    uint32_t& key     = *std::bit_cast<uint32_t*>(keyAddr);
                    key               = keyHash;

                    uintptr_t refCounterAddr = getAtomicCounterAddress(elemAddr, m_sNodeSize, m_paddingEnd);
                    auto*     refCounter     = std::bit_cast<std::atomic<uint32_t>*>(refCounterAddr);
                    refCounter->store(0, std::memory_order_release);

                    uint64_t valueSize = getValueSize();
                    std::memcpy(std::bit_cast<void*>(elemAddr), pValue, valueSize);

                    m_size.fetch_add(1, std::memory_order_release);

                    inserted = true;
                    unlockForWrite(sNode, i);
                    break;
                }

                unlockForAlloc(pNode, packet, 0b00);

                if (!inserted)
                {
                    while (!lockForAlloc(pNode, packet, true))
                        std::this_thread::yield();

                    TaggedPointer& childPtr    = pNode->children[packet];
                    void*          childRawPtr = m_table.rawPtr(childPtr);

                    assert(!isINode(pNode, packet));
                    assert(childRawPtr);

                    SNode* oldSNode = std::bit_cast<SNode*>(childRawPtr);

                    childPtr        = m_table.allocate(mctx, sizeof(INode), alignof(INode));
                    INode* newINode = std::bit_cast<INode*>(m_table.rawPtr(childPtr));

                    std::memset(newINode->children, 0, sizeof(newINode->children));
                    newINode->bitmap.store(std::numeric_limits<uint64_t>::max() >> 4, std::memory_order_acq_rel);

                    uint32_t firstPacket            = (keyHash & mask) >> (shamt - 5);
                    newINode->children[firstPacket] = m_table.allocate(mctx, sizeof(SNode), alignof(SNode));
                    std::memcpy(std::bit_cast<void*>(m_table.rawPtr(newINode->children[firstPacket])),
                                oldSNode,
                                sizeof(SNode));

                    uint32_t secondPacket            = (keyHash & mask) >> (shamt - 5);
                    newINode->children[secondPacket] = m_table.allocate(mctx, sizeof(SNode), alignof(SNode));
                    SNode* newChildSNode = std::bit_cast<SNode*>(m_table.rawPtr(newINode->children[secondPacket]));

                    std::memset(newChildSNode->data, 0, sizeof(newChildSNode->data));
                    newChildSNode->bitmap.store(0, std::memory_order_release);

                    unlockForAlloc(pNode, packet, 0b10);

                    return iinsert(mctx, std::bit_cast<INode*>(newChildSNode), keyHash, pValue);
                }

                return EResult::eOk;
            }
        }
    }

    bool CTrie::lockForAlloc(INode* self, uint32_t childIdx, bool force)
    {
        // Read the bitmap for the slot
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);

        // Shift to the correct position for the element in the bitmap
        uint32_t shamt = childIdx << 1;
        uint64_t mask  = 0b11 << shamt;

        uint64_t bits = (bitmap & mask) >> shamt;

        // if somebody already allocated this in the meantime, bail out
        if (!force && bits != 0b11)
        {
            // if the allocation is still in progress, wait
            while (bits == 0b01)
            {
                std::this_thread::yield();

                // Re-read the bitmap to get the latest state
                bitmap = self->bitmap.load(std::memory_order_acquire);
                bits   = (bitmap & mask) >> shamt;
            }

            return false;
        }

        // Attempt to lock the slot for "allocation in progress"
        uint64_t desired = (bitmap & ~mask) | (0b01 << shamt); // Set to 0b10 (locked for allocation in progress)

        // Compare-and-swap to atomically update the bitmap
        bool success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_seq_cst, std::memory_order_acquire);

        // If the CAS failed, retry until successful
        if (!success)
        {
            return lockForAlloc(self, childIdx);
        }
        else
        {
            return true;
        }
    }

    void CTrie::unlockForAlloc(INode* self, uint32_t childIdx, uint64_t desired)
    {
        assert(desired == 0b00 || desired == 0b10); // should be either SNode or INode

        // Read the bitmap for the slot
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);

        // Shift to the correct position for the element in the bitmap
        uint32_t shamt = childIdx << 1;
        uint64_t mask  = 0b11 << shamt;

        uint64_t bits = (bitmap & mask) >> shamt;
        assert(bits == 0b01); // an allocation should be in progress

        // Attempt to lock the slot for "allocation in progress"
        desired = (bitmap & ~mask) | (desired << shamt);
        bool success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_seq_cst, std::memory_order_seq_cst);
    }

    bool CTrie::isINode(INode const* parent, uint32_t childIdx)
    {
        assert(childIdx < 32);
        uint64_t bitmap = parent->bitmap.load(std::memory_order_acquire);
        uint32_t shamt  = childIdx << 1;
        uint64_t bits   = (bitmap & (0b11 << shamt)) >> shamt;
        return bits == 0b10;
    }

    bool CTrie::isReadLockable(SNode const* self, uint32_t elementIdx)
    {
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);
        uint32_t shamt  = elementIdx << 1;
        uint64_t bits   = (bitmap & (0b11 << shamt)) >> shamt;
        return bits == 0b01 || bits == 0b11;
    }

    bool CTrie::isFullElement(SNode const* self, uint32_t elementIdx)
    {
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);
        uint32_t shamt  = elementIdx << 1;
        uint64_t bits   = (bitmap & (0b11 << shamt)) >> shamt;
        return bits != 0b00;
    }

    bool CTrie::isFree(SNode const* self, uint32_t elementIdx)
    {
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);
        uint32_t shamt  = elementIdx << 1;
        uint64_t bits   = (bitmap & (0b11 << shamt)) >> shamt;
        return bits == 0b00;
    }

    bool CTrie::isChildNotAllocated(INode const* parent, uint32_t childIdx)
    {
        assert(childIdx < 32);
        uint64_t bitmap = parent->bitmap.load(std::memory_order_acquire);
        uint32_t shamt  = childIdx << 1;
        uint64_t bits   = (bitmap & (0b11 << shamt)) >> shamt;
        return bits == 0b11;
    }

    void CTrie::lockForWrite(SNode* self, uint32_t elementIdx)
    {
        // Read the bitmap for the slot
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);

        // Shift to the correct position for the element in the bitmap
        uint32_t shamt = elementIdx << 1;
        uint64_t mask  = 0b11 << shamt;

        uint64_t bits = (bitmap & mask) >> shamt;

        // Keep checking until it is not locked for read or write (0b10 or 0b01)
        while (bits == 0b10 || bits == 0b01)
        {
            // Yield to allow other threads to proceed
            std::this_thread::yield();

            // Re-read the bitmap to get the latest state
            bitmap = self->bitmap.load(std::memory_order_acquire);
            bits   = (bitmap & mask) >> shamt;
        }

        // Attempt to lock the slot for writing (set the bits to 0b10)
        uint64_t desired = (bitmap & ~mask) | (0b10 << shamt); // Set to 0b10 (locked for write)

        // Compare-and-swap to atomically update the bitmap
        bool success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_seq_cst, std::memory_order_acquire);

        // If the CAS failed, retry until successful
        if (!success)
        {
            lockForWrite(self, elementIdx);
        }
    }

    void CTrie::lockForRead(SNode* self, uint32_t elementIdx, size_t nodeSize, uint32_t paddingEnd)
    {
        // Read the bitmap for the slot
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);

        // Shift to the correct position for the element in the bitmap
        uint32_t shamt = elementIdx << 1;
        uint64_t mask  = 0b11 << shamt;

        uint64_t bits = (bitmap & mask) >> shamt;
        assert(bits != 0b00); // should be alread allocated and not null

        // Keep checking until it is not write locked
        while (bits == 0b10)
        {
            // Yield to allow other threads to proceed
            std::this_thread::yield();

            // Re-read the bitmap to get the latest state
            bitmap = self->bitmap.load(std::memory_order_acquire);
            bits   = (bitmap & mask) >> shamt;
        }

        // Attempt to lock the slot for writing (set the bits to 0b10)
        uint64_t desired = (bitmap & ~mask) | (0b01 << shamt); // Set to 0b01 (locked for read)

        // Compare-and-swap to atomically update the bitmap
        bool success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_seq_cst, std::memory_order_acquire);

        // If the CAS failed, retry until successful
        if (!success)
        {
            lockForRead(self, elementIdx, nodeSize, paddingEnd);
        }

        // Get the reference counter and increment
        void const* data           = self->data;
        uintptr_t   elemAddr       = std::bit_cast<uintptr_t>(data);
        uintptr_t   refCounterAddr = getAtomicCounterAddress(elemAddr, nodeSize, paddingEnd);
        auto*       refCounter     = std::bit_cast<std::atomic<uint32_t>*>(refCounterAddr);

        uint32_t count = refCounter->fetch_add(1, std::memory_order_acquire);
    }

    void CTrie::unlockForWrite(SNode* self, uint32_t elementIdx)
    {
        // Read the bitmap for the slot
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);

        // Shift to the correct position for the element in the bitmap
        uint32_t shamt = elementIdx << 1;
        uint64_t mask  = 0b11 << shamt;

        uint64_t bits = (bitmap & mask) >> shamt;

        // Ensure that the slot is currently locked for write (0b10)
        assert(bits == 0b10);

        // Change the bitmap to set the element back to 0b01 (read-lockable)
        uint64_t desired = (bitmap & ~mask) | (0b11 << shamt); // Set to 0b11 (unused)

        // Atomically update the bitmap using compare-and-swap
        bool success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_acq_rel, std::memory_order_acquire);

        // If the CAS failed, retry until successful
        if (!success)
        {
            unlockForWrite(self, elementIdx);
        }
    }

    void CTrie::unlockForRead(SNode* self, uint32_t elementIdx, size_t nodeSize, uint32_t paddingEnd)
    {
        // Read the bitmap for the slot
        uint64_t bitmap = self->bitmap.load(std::memory_order_acquire);

        // Shift to the correct position for the element in the bitmap
        uint32_t shamt = elementIdx << 1;
        uint64_t mask  = 0b11 << shamt;

        uint64_t bits = (bitmap & mask) >> shamt;

        // Ensure that the slot is currently locked for read (0b01)
        assert(bits == 0b01);

        // decrement reference counter
        void const* data           = self->data;
        uintptr_t   elemAddr       = std::bit_cast<uintptr_t>(data);
        uintptr_t   refCounterAddr = getAtomicCounterAddress(elemAddr, nodeSize, paddingEnd);
        auto*       refCounter     = std::bit_cast<std::atomic<uint32_t>*>(refCounterAddr);

        uint32_t count = refCounter->fetch_sub(1, std::memory_order_release);
        assert(count != 0);

        // free the node
        uint64_t desired = (bitmap & ~mask) | (0b11 << shamt); // Set to 0b11 (unused)
        bool success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_acq_rel, std::memory_order_acquire);
        // If the CAS failed, retry until successful
        while (!success)
        {
            success = self->bitmap.compare_exchange_strong(bitmap, desired, std::memory_order_acq_rel, std::memory_order_acquire);
        }
    }

    // Parsing --------------------------------------------------------------------------------------------------------
    // https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/parser.cpp
    char WordParser::decodeEscaped(char c)
    {
        switch (c)
        {
            case EOF:
                // You shouldn't be here
                return EOF;
            case 'b':
                return '\b';
            case 'f':
                return '\f';
            case 'n':
                return '\n';
            case 'r':
                return '\r';
            case 't':
                return '\t';
            case '\\':
                return '\\';
            case '\'':
                return '\'';
            case '\"':
                return '\"';
            default:
                assert(false && "invalid escaped character");
                std::abort();
        }
        return 0; // NOTREACHED
    }

    char WordParser::getChar(std::string_view str, size_t idx)
    {
        if (m_needsContinuation && idx < m_bufferLength)
        {
            return m_buffer[idx];
        }
        else if (m_needsContinuation)
        {
            return str[idx - m_bufferLength];
        }
        else
        {
            return str[idx];
        }
    }

    void WordParser::copyToBuffer(std::string_view str)
    {
        assert(m_bufferLength + str.size() < 256);
        std::memcpy(m_buffer + m_bufferLength, str.data(), str.size());
        m_bufferLength += static_cast<uint32_t>(str.size());
    }

    std::string_view WordParser::catResult(std::string_view str, size_t start, size_t end)
    {
        if (m_needsContinuation)
        {
            m_needsContinuation = false;
            if (end > start + m_bufferLength) // if you read past the buffer
            {
                size_t len = end - start - m_bufferLength;
                assert(m_bufferLength + len < 256);
                m_numCharReadLastTime += len;
                std::string_view s = str.substr(0, len);
                std::memcpy(m_buffer + m_bufferLength, s.data(), s.size());
                m_bufferLength += static_cast<uint32_t>(s.size());
            }
            else // you read only the m_buffer (the whole thing, so no need to change buffer langth)
            {
                m_numCharReadLastTime = 0;
            }
        }
        else
        {
            assert(end > start);
            assert(end - start <= 256);
            size_t len = end - start;
            m_numCharReadLastTime += len;
            copyToBuffer(str.substr(start, len));
        }

        return {m_buffer, m_bufferLength};
    }

    bool WordParser::needsContinuation() const
    {
        return m_needsContinuation;
    }

    uint32_t WordParser::numCharReadLast() const
    {
        return m_numCharReadLastTime;
    }

    bool WordParser::endOfStr(std::string_view str, size_t idx) const
    {
        if (m_needsContinuation)
        {
            if (idx >= m_bufferLength)
                idx = idx - m_bufferLength;
            else
                idx = 0ULL;
        }

        return idx >= str.size();
    }

    std::string_view WordParser::nextWord(std::string_view str)
    {
        if (!m_needsContinuation)
        {
            std::memset(m_buffer, 0, sizeof(m_buffer));
            std::memset(m_escapedBuffer, 0, sizeof(m_escapedBuffer));
            m_bufferLength = 0;
        }
        m_numCharReadLastTime = 0;

        size_t i = 0;
        while (i < str.size())
        {
            size_t start = i;
            char   c     = getChar(str, i++);

            if (std::isspace(static_cast<unsigned char>(c)))
            { // nothing
                ++m_numCharReadLastTime;
            }
            else if (c == '"') // parse string, scan to closing quote
            {
                if (!m_needsContinuation)
                {
                    m_haveEscaped = false;
                }

                while (true)
                {
                    if (endOfStr(str, i))
                    { // string parsing was interrupted by the end of the chunk
                        copyToBuffer(str.substr(start, i - start));
                        m_needsContinuation = true;
                        return "";
                    }

                    if ((c = getChar(str, i++)) != '"')
                    {
                        if (c == '\n')
                        { // TODO error hendling
                            m_needsContinuation = false;
                            return "";
                        }
                        else if (c == '\\')
                        {
                            m_haveEscaped = true;
                        }
                    }
                    else
                    {
                        break;
                    }
                } // while not end quote

                if (!m_haveEscaped)
                {
                    return catResult(str, start, i);
                }
                else
                { // TODO decude escaped
                    m_haveEscaped   = false;
                    uint32_t escIdx = 0;
                    for (uint32_t j = start; j < i; ++j)
                    {
                        if (getChar(str, j) != '\\')
                        {
                            m_escapedBuffer[escIdx++] = str[j];
                        }
                        else
                        {
                            ++j; // go past '\\'
                            assert(j < i);
                            m_escapedBuffer[escIdx++] = decodeEscaped(str[j]);
                        }
                        m_escapedBuffer[escIdx] = '\0';
                        return catResult({m_escapedBuffer, escIdx}, start, i);
                    }
                }
            } // end parse string
            else if (c == '[' || c == ']') // parse begin/end array
            {
                m_needsContinuation = false;
                return catResult(str, start, start + 1);
            }
            else if (c == '#') // comment. Scan until EOL or EOF
            {
                while (true)
                {
                    if (endOfStr(str, i))
                    {
                        copyToBuffer(str.substr(start, i - start));
                        m_needsContinuation = true;
                        return "";
                    }

                    c = getChar(str, i++);
                    if (c == '\n' || c == '\r')
                    {
                        --i;
                        break;
                    }
                }

                return catResult(str, start, i);
            }
            else // regular character. go until end of word/number
            {
                while (true)
                {
                    if (endOfStr(str, i))
                    {
                        copyToBuffer(str.substr(start, i - start));
                        m_needsContinuation = true;
                        return "";
                    }

                    c = getChar(str, i++);
                    if (std::isspace(static_cast<unsigned char>(c)) || c == '"' || c == '[' || c == ']')
                    {
                        --i;
                        break;
                    }
                }

                return catResult(str, start, i);
            }
        }

        // you found only whitespaces
        m_needsContinuation = false;
        return "";
    }

    bool HeaderTokenizer::parseNext(std::string_view* pChunk, size_t& inOutoffset)
    {
        if (!pChunk || pChunk->empty())
        {
            return false;
        }

        std::string_view& chunk  = *pChunk;
        size_t            offset = inOutoffset;

        // skip whitespace and comments
        while (offset < chunk.size())
        {
            char c = chunk[offset];
            if (c == '#' || c == '\n')
            {
                while (offset < chunk.size() && chunk[offset] != '\n')
                    ++offset;
            }
            else if (std::isspace(c))
            {
                ++offset;
            }
            else // found a meaningful character
            {
                break;
            }
        }

        inOutoffset = offset; // update offset before attempting parsing

        if (offset >= chunk.size())
        { // end of chunk; processing continues in the next chunk
            return false;
        }

        // TODO remove
        return false;
    }

    void HeaderTokenizer::advance()
    {
        if (m_finished)
        {
            return;
        }
        std::string_view* pChunk          = nullptr;
        size_t            effectiveOffset = 0;
        size_t            advancedBy      = 0;
        bool              tokenReady      = false;
        std::memset(m_storage.bytes.data(), 0, m_storage.bytes.size());

        if (!m_useCurr)
        {
            pChunk          = &m_prevChunk;
            effectiveOffset = m_offset + m_prevOffset;
            tokenReady      = parseNext(pChunk, effectiveOffset);
        }

        if (!tokenReady)
        {
            pChunk          = &m_currChunk;
            effectiveOffset = m_offset - (m_prevChunk.size() - m_prevOffset);
            tokenReady      = parseNext(pChunk, effectiveOffset);
        }

        m_started = true;
    }

    bool HeaderTokenizer::hasToken() const
    {
        return m_started && !m_finished;
    }

    HeaderTokenizer::Storage HeaderTokenizer::retrieveToken(EHeaderTokenType& outTokenType) const
    {
        assert(m_started && !m_finished);
        assert(m_currentToken != EHeaderTokenType::eCount);
        outTokenType = m_currentToken;
        return m_storage;
    }

    size_t HeaderTokenizer::offsetFromCurrent() const
    {
        assert(m_offset > m_prevOffset);
        assert(m_finished);
        return m_offset - m_prevOffset;
    }
} // namespace dmt

namespace dmt::model {
} // namespace dmt::model

namespace dmt::job {
    void parseSceneHeader(uintptr_t address)
    {
        using namespace dmt;
        char                  buffer[512]{};
        ParseSceneHeaderData& data = *std::bit_cast<ParseSceneHeaderData*>(address);
        AppContext&           actx = *data.actx;
        actx.log("Starting Parse Scene Header Job");
        bool error = false;

        ChunkedFileReader reader{actx.mctx.pctx, data.filePath.data(), 512};
        if (reader)
        {
            for (uint32_t chunkNum = 0; chunkNum < reader.numChunks(); ++chunkNum)
            {
                bool status = reader.requestChunk(actx.mctx.pctx, buffer, chunkNum);
                if (!status)
                {
                    error = true;
                    break;
                }

                status = reader.waitForPendingChunk(actx.mctx.pctx);
                if (!status)
                {
                    error = true;
                    break;
                }

                uint32_t         size = reader.lastNumBytesRead();
                std::string_view chunkView{buffer, size};
                actx.log("Read chunk content:\n{}\n", {chunkView});
            }
        }
        else
        {
            actx.error("Couldn't open file \"{}\"", {data.filePath});
        }

        if (error)
        {
            actx.error("Something went wrong during job execution");
        }

        actx.log("Parse Scene Header Job Finished");
        std::atomic_thread_fence(std::memory_order_release);
        std::atomic_store_explicit(&data.done, 1, std::memory_order_relaxed);
    }
} // namespace dmt::job