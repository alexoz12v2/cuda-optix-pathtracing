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
            assert(m_bufferLength + str.size() < 256);
            assert(end > start + m_bufferLength);
            m_numCharReadLastTime += end - start - m_bufferLength;
            m_needsContinuation = false;
            std::string_view s  = str.substr(0, m_numCharReadLastTime);
            std::memcpy(m_buffer + m_bufferLength, s.data(), s.size());
            m_bufferLength += static_cast<uint32_t>(s.size());
        }
        else
        {
            assert(end > start);
            m_numCharReadLastTime += end - start;
            copyToBuffer(str.substr(start, m_numCharReadLastTime));
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
                            if ((c = getChar(str, i++)) == EOF)
                            { // the string was interrupded by the end of the chunk
                                copyToBuffer(str.substr(start, i - start));
                                m_needsContinuation = true;
                                return "";
                            }
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
                return catResult(str, start, start + 1);
            }
            else if (c == '#') // comment. Scan until EOL or EOF
            {
                while ((c = getChar(str, i++)) != EOF)
                {
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