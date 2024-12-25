module;

#include <utility>

#include <cstdint>
#include <cstdlib>

module platform;

namespace dmt {
    AppContext::AppContext(void*                                      platformContextData,
                           PlatformContext::Table const*              pTable,
                           PlatformContext::InlineTable const&        inlineTable,
                           uint32_t                                   pageTrackCapacity,
                           uint32_t                                   allocTrackCapacity,
                           std::array<uint32_t, numBlockSizes> const& numBlocksPerPool) :
    mctx{platformContextData, pTable, inlineTable, pageTrackCapacity, allocTrackCapacity, numBlocksPerPool},
    threadPool{mctx}
    {
    }

    void AppContext::write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
    {
        mctx.pctx.write(level, str, loc);
    }

    void AppContext::write(ELogLevel                            level,
                           std::string_view const&              str,
                           std::initializer_list<StrBuf> const& list,
                           std::source_location const&          loc)
    {
        mctx.pctx.write(level, str, list, loc);
    }

    AppContext::~AppContext()
    {
        threadPool.cleanup(mctx);
        mctx.cleanup();
    }

    Platform::Platform()
    {
    }

    Platform::Platform(Platform&&) noexcept
    {
    }

    uint64_t Platform::getSize() const
    {
        return 4096;
    }

    Platform& Platform::operator=(Platform&&) noexcept
    {
        return *this;
    }

    Platform::~Platform() noexcept
    {
    }

} // namespace dmt