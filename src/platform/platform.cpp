#define DMT_INTERFACE_AS_HEADER
<<<<<<< HEAD
#include <platform.h>
=======
#include "platform.h"

>>>>>>> 7927d283b66d090ebdc7959fc958c0785a900c9f
#include <utility>

#include <cstdint>
#include <cstdlib>

<<<<<<< HEAD
//module platform;


=======
>>>>>>> 7927d283b66d090ebdc7959fc958c0785a900c9f
namespace dmt {
    AppContext::AppContext(uint32_t                                   pageTrackCapacity,
                           uint32_t                                   allocTrackCapacity,
                           std::array<uint32_t, numBlockSizes> const& numBlocksPerPool) :
    mctx{pageTrackCapacity, allocTrackCapacity, numBlocksPerPool},
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

    size_t AppContext::maxLogArgBytes() const { return mctx.pctx.maxLogArgBytes(); }

    void AppContext::addJob(Job const& job, EJobLayer layer) { threadPool.addJob(mctx, job, layer); }

    AppContext::~AppContext()
    {
        threadPool.cleanup(mctx);
        mctx.cleanup();
    }

    Platform::Platform() {}

    Platform::Platform(Platform&&) noexcept {}

    uint64_t Platform::getSize() const { return 4096; }

    Platform& Platform::operator=(Platform&&) noexcept { return *this; }

    Platform::~Platform() noexcept {}

} // namespace dmt