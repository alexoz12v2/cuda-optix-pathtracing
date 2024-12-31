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

    std::atomic<uint32_t>& SNode::refCounterAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        // Compute the base address of the element.
        uintptr_t elemBaseAddr = reinterpret_cast<uintptr_t>(data.data()) + index * valueSize;

        // Ensure alignment of the base address.
        uint64_t  mask            = valueAlign - 1;
        uintptr_t alignedElemAddr = (elemBaseAddr + mask) & ~mask;
        assert(elemBaseAddr == alignedElemAddr);
        assert(alignedElemAddr + sizeof(std::atomic<uint32_t>) <= reinterpret_cast<uintptr_t>(data.data()) + data.size());

        // Calculate the offset for the reference counter within the structure.
        uintptr_t refCounterAddr = alignToAddr(alignedElemAddr + valueSize, alignof(std::atomic<uint32_t>));

        return *reinterpret_cast<std::atomic<uint32_t>*>(refCounterAddr);
    }

    uint32_t SNode::incrRefCounter(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        return refCounterAt(index, valueSize, valueAlign).fetch_add(1, std::memory_order_seq_cst) + 1;
    }

    uint32_t SNode::decrRefCounter(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        return refCounterAt(index, valueSize, valueAlign).fetch_sub(1, std::memory_order_seq_cst) - 1;
    }

    uint32_t& SNode::keyRef(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        uintptr_t elemBaseAddr = reinterpret_cast<uintptr_t>(data.data()) + index * valueSize;

        // Ensure alignment of the base address.
        uint64_t  mask            = valueAlign - 1;
        uintptr_t alignedElemAddr = (elemBaseAddr + mask) & ~mask;
        assert(elemBaseAddr == alignedElemAddr);
        assert(alignedElemAddr + sizeof(uint32_t) <= reinterpret_cast<uintptr_t>(data.data()) + data.size());

        // Calculate the offset for the key within the structure.
        uintptr_t refCounterAddr = alignToAddr(alignedElemAddr + valueSize, alignof(std::atomic<uint32_t>));
        uintptr_t keyAddr        = refCounterAddr + sizeof(uint32_t);

        return *reinterpret_cast<uint32_t*>(keyAddr);
    }

    uintptr_t SNode::getElementBaseAddress(uint32_t index, uint32_t valueSize, uint32_t valueAlign) const
    {
        uintptr_t elemBaseAddr = reinterpret_cast<uintptr_t>(data.data()) + index * valueSize;

        // Ensure alignment of the base address
        uint64_t  mask            = valueAlign - 1;
        uintptr_t alignedElemAddr = (elemBaseAddr + mask) & ~mask;
        assert(elemBaseAddr == alignedElemAddr);
        assert(alignedElemAddr + valueSize <= reinterpret_cast<uintptr_t>(data.data()) + data.size());

        return alignedElemAddr;
    }

    void const* SNode::valueConstAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign) const
    {
        uintptr_t baseAddr = getElementBaseAddress(index, valueSize, valueAlign);
        return reinterpret_cast<void const*>(baseAddr);
    }

    // Access value as void*
    void* SNode::valueAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        uintptr_t baseAddr = getElementBaseAddress(index, valueSize, valueAlign);
        return reinterpret_cast<void*>(baseAddr);
    }

    TaggedPointer CTrie::newINode(MemoryContext& mctx) const
    {
        TaggedPointer ret   = m_table.allocate(mctx, sizeof(INode), alignof(INode));
        auto*         pNode = reinterpret_cast<SNode*>(m_table.rawPtr(ret));
        std::construct_at(pNode);
        return ret;
    }

    TaggedPointer CTrie::newSNode(MemoryContext& mctx) const
    {
        TaggedPointer ret   = m_table.allocate(mctx, sizeof(INode), alignof(INode));
        auto*         pNode = reinterpret_cast<INode*>(m_table.rawPtr(ret));
        std::construct_at(pNode);
        return ret;
    }

    CTrie::CTrie(MemoryContext& mctx, AllocatorTable const& table, uint32_t valueSize, uint32_t valueAlign) :
    m_table(table),
    m_root(newSNode(mctx)),
    m_valueSize(valueSize),
    m_valueAlign(valueAlign)
    {
        if (m_root == taggedNullptr)
        {
            mctx.pctx.error("Couldn't allocate root node for CTrie");
            std::abort();
        }
    }

    inline constexpr uint32_t posINode(uint32_t hash, uint32_t level)
    {
        uint32_t index = (hash >> level) & (cardinalityINode - 1);
        return index;
    }

    inline constexpr uint32_t elemSize(uint32_t valueSize, uint32_t valueAlign)
    { 
        // 2. Padding to ensure valueAlign alignment for the ref counter + key.
        uint32_t alignedOffset = (valueSize + valueAlign - 1) & ~(valueAlign - 1);

        // 3. Atomic ref counter + key (uint32_t + uint32_t = 8 bytes).
        constexpr uint32_t metadataSize = sizeof(std::atomic<uint32_t>) + sizeof(uint32_t);

        // Calculate the total element size.
        uint32_t elementSize = alignedOffset + metadataSize;
        return elementSize;
    }

    inline constexpr uint32_t posSNode(uint32_t hash, uint32_t level, uint32_t valueSize, uint32_t valueAlign)
    {
        uint32_t elementSize = elemSize(valueSize, valueAlign);

        // Calculate the number of elements that can fit.
        constexpr uint32_t dataSize    = sizeof(SNode::Data); // SNode data size
        uint32_t           numElements = dataSize / elementSize;

        // Calculate the position.
        uint32_t pos = (hash >> level) & smallestPOTMask(numElements);

        return clamp(pos, 0u, numElements - 1u);
    }

    void CTrie::cleanupINode(MemoryContext& mctx, INode* inode, void(*dctor)(MemoryContext& mctx, void* value))
    {
        for (uint32_t i = 0; i < cardinalityINode; ++i)
        {
            while (inode->bits.checkOccupied(i))
                std::this_thread::yield();

            if (inode->bits.checkFree(i))
                continue;

            else if (inode->bits.checkINode(i))
            {
                while (!inode->bits.setOccupied(i, true))
                    std::this_thread::yield();

                TaggedPointer pt = inode->children[i];
                INode* child = reinterpret_cast<INode*>(m_table.rawPtr(pt));
                cleanupINode(mctx, child, dctor);
                m_table.free(mctx, pt, sizeof(INode), alignof(INode));
            }
            else if (inode->bits.checkSNode(i))
            {
                while (!inode->bits.setOccupied(i, true))
                    std::this_thread::yield();

                TaggedPointer pt = inode->children[i];
                SNode* snode = reinterpret_cast<SNode*>(m_table.rawPtr(pt));
                cleanupSNode(mctx, snode, dctor);
                m_table.free(mctx, pt, sizeof(SNode), alignof(SNode));
            }
            else
                assert(false);
        }
    }

    void CTrie::cleanupSNode(MemoryContext& mctx, SNode* snode, void(*dctor)(MemoryContext& mctx, void* value))
    {
        uint32_t elementSize = elemSize(m_valueSize, m_valueAlign);
        uint32_t capacity    = sizeof(SNode::Data) / elementSize;
        for (uint32_t i = 0; i < capacity; ++i)
        {
            if (snode->bits.checkFree(i))
                continue;
            while (snode->bits.checkWriteLocked(i))
                std::this_thread::yield();
            if (!snode->bits.casFreeToWriteLocked(i, SNodeBitmap::unused))
                continue;

            void* elem = snode->valueAt(i, m_valueSize, m_valueAlign);
            dctor(mctx, elem);
            snode->bits.setFree(i);
        }
    }

    void CTrie::cleanup(MemoryContext& mctx, void(*dctor)(MemoryContext& mctx, void* value))
    {
        INode* pRoot = reinterpret_cast<INode*>(m_table.rawPtr(m_root));
        cleanupINode(mctx, pRoot, dctor);
        m_table.free(mctx, m_root, sizeof(INode), alignof(INode));
    }

    bool CTrie::insert(MemoryContext& mctx, uint32_t key, void const* value)
    {
        INode*   inode = reinterpret_cast<INode*>(m_table.rawPtr(m_root));
        SNode*   snode = nullptr;
        uint32_t level = 0;
        while (level < 5)
        {
            if (!snode) // current node is inode
            {
                uint32_t pos = posINode(key, level);
                if (inode->bits.checkFree(pos) && inode->bits.setOccupied(pos))
                { // allocate a new snode
                    TaggedPointer ptr = newSNode(mctx);
                    if (ptr == taggedNullptr)
                    {
                        inode->bits.setFree(pos);
                        return false;
                    }
                    inode->children[pos] = ptr;
                    inode->bits.setSNode(pos);
                }
                else
                {
                    while (inode->bits.checkOccupied(pos))
                        std::this_thread::yield();
                    if (inode->bits.checkFree(pos))
                        return false;
                }

                if (inode->bits.checkINode(pos))
                    inode = reinterpret_cast<INode*>(m_table.rawPtr(inode->children[pos]));
                else if (inode->bits.checkSNode(pos))
                    snode = reinterpret_cast<SNode*>(m_table.rawPtr(inode->children[pos]));
                else
                {
                    assert(false);
                    return false;
                }
            }
            else
            {
                uint32_t pos = posSNode(key, level, m_valueSize, m_valueAlign);
                while (snode->bits.checkWriteLocked(pos))
                    std::this_thread::yield();
                if (snode->bits.checkFree(pos) && snode->bits.casFreeToWriteLocked(pos))
                { // write the value inside the thing
                    std::memcpy(snode->valueAt(pos, m_valueSize, m_valueAlign), value, m_valueSize);
                    snode->keyRef(pos, m_valueSize, m_valueAlign) = key;
                    snode->refCounterAt(pos, m_valueSize, m_valueAlign).store(0, std::memory_order_seq_cst);
                    snode->bits.setUnused(pos);
                    return true;
                }
                else
                { // TODO collision! 
                    return false;
                }
            }

            ++level;
        }

        return false;
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