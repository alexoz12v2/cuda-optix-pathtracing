#pragma once

// keep in sync with .cppm file
#include "dmtmacros.h"
#include "platform/platform-macros.h"
#include "platform/platform-mixed-macros.h"

#include <platform/platform-utils.h>
#include <platform/platform-logging.h>
#include <platform/platform-memory.h>
#include <platform/platform-threadPool.h>
#include <platform/platform-display.h>

#if !defined(DMT_NEEDS_MODULE)
#include <array>
#include <bit>
#include <charconv>
#include <exception>
#include <iterator>
#include <memory>
#include <memory_resource>
#include <source_location>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <cctype>
#include <cmath>
#include <cstdint>
#endif

namespace dmt {
    class AppContext;
    namespace ctx {
        using namespace dmt;
        DMT_PLATFORM_MIXED_API void        init(AppContext& ctx);
        DMT_PLATFORM_MIXED_API AppContext* acquireCurrent();
        DMT_PLATFORM_MIXED_API void        releaseCurrent();
        DMT_PLATFORM_MIXED_API void        unregister();
    } // namespace ctx

    class DMT_PLATFORM_MIXED_API AppContextJanitor
    {
    public:
        AppContextJanitor() : actx(*ctx::acquireCurrent()) {}
        AppContextJanitor(AppContextJanitor const&)                = delete;
        AppContextJanitor(AppContextJanitor&&) noexcept            = delete;
        AppContextJanitor& operator=(AppContextJanitor const&)     = delete;
        AppContextJanitor& operator=(AppContextJanitor&&) noexcept = delete;
        ~AppContextJanitor() { ctx::releaseCurrent(); }

        AppContext& actx;
    };

    struct AppContextImpl;
    class DMT_PLATFORM_MIXED_API AppContext : public InterfaceLogger<AppContext>
    {
        friend DMT_PLATFORM_MIXED_API void ctx::init(AppContext& ctx);
        friend DMT_PLATFORM_MIXED_API void ctx::unregister();

    public:
        AppContext(uint32_t                                   pageTrackCapacity,
                   uint32_t                                   allocTrackCapacity,
                   std::array<uint32_t, numBlockSizes> const& numBlocksPerPool);
        AppContext(AppContext const&);
        AppContext(AppContext&&) noexcept;
        AppContext& operator=(AppContext const&);
        AppContext& operator=(AppContext&&) noexcept;
        ~AppContext() noexcept;

    public:
        // logging methods
        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc);
        void write(ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc);

        bool     enabled(ELogLevel level) const;
        bool     traceEnabled() const;
        bool     logEnabled() const;
        bool     warnEnabled() const;
        bool     errorEnabled() const;
        void     dbgTraceStackTrace();
        void     dbgErrorStackTrace();
        size_t   maxLogArgBytes() const;
        uint64_t millisFromStart() const;

        // threadpool methods
        void addJob(Job const& job, EJobLayer layer);
        void kickJobs();
        void pauseJobs();
        bool otherLayerActive(EJobLayer& layer) const;

        // memory methods
        void*                   stackAllocate(size_t size, size_t alignment, EMemoryTag tag, sid_t sid);
        void                    stackReset();
        TaggedPointer           poolAllocateBlocks(uint32_t numBlocks, EBlockSize blockSize, EMemoryTag tag, sid_t sid);
        void                    poolFreeBlocks(uint32_t numBlocks, TaggedPointer ptr);
        PageAllocationsTracker& tracker();

        // string table
        sid_t            intern(std::string_view str);
        std::string_view lookup(sid_t sid);

        // Necessary evil for `cudaHello` function. don't use this
        MemoryContext& mctx();

    private:
        void cleanup() noexcept;

    private:
        AppContextImpl* m_pimpl;
    };
    static_assert(sizeof(AppContext) == sizeof(uintptr_t));
} // namespace dmt
