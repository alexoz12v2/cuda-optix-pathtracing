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

DMT_MODULE_EXPORT namespace dmt {
    class AppContext;
    namespace ctx {
        using namespace dmt;
        DMT_PLATFORM_MIXED_API void        init(AppContext& ctx);
        DMT_PLATFORM_MIXED_API AppContext* acquireCurrent();
        DMT_PLATFORM_MIXED_API void        releaseCurrent();
        DMT_PLATFORM_MIXED_API void        unregister();
    }

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
        size_t maxLogArgBytes() const;
        void   write(ELogLevel level, std::string_view const& str, std::source_location const& loc);
        void   write(ELogLevel                            level,
                     std::string_view const&              str,
                     std::initializer_list<StrBuf> const& list,
                     std::source_location const&          loc);
        void   addJob(Job const& job, EJobLayer layer);

    private:
        void cleanup() noexcept;

    private:
        AppContextImpl* m_pimpl;
    };
    static_assert(sizeof(AppContext) == sizeof(uintptr_t));
} // namespace dmt
