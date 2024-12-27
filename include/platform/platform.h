#pragma once

#include "dmtmacros.h"

#include <array>
#include <source_location>
#include <string_view>

#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-logging.h>
#include <platform/platform-memory.h>
#include <platform/platform-threadPool.h>
#include <platform/platform-utils.h>
#else
import <platform/platform-logging.h>;
import <platform/platform-memory.h>;
import <platform/platform-utils.h>;
import <platform/platform-threadPool.h>;
#endif

DMT_MODULE_EXPORT dmt {
    struct AppContext : public InterfaceLogger<AppContext>
    {
        AppContext(uint32_t                                   pageTrackCapacity,
                   uint32_t                                   allocTrackCapacity,
                   std::array<uint32_t, numBlockSizes> const& numBlocksPerPool);
        AppContext(AppContext const&)                = delete;
        AppContext(AppContext&&) noexcept            = delete;
        AppContext& operator=(AppContext const&)     = delete;
        AppContext& operator=(AppContext&&) noexcept = delete;
        ~AppContext();

        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc);
        void write(ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc);

        MemoryContext mctx;
        ThreadPoolV2  threadPool;
    };

    // TODO boot up request sudo access
    // TODO log level should be owned by the Platform class only
    /**
     * @class Platform
     * @brief Class whose constructor initializes all the necessary objects to bootstrap the application
     */
    class Platform
    {
    public:
        Platform();
        Platform(Platform const&) = delete;
        Platform(Platform&&) noexcept;
        Platform& operator=(Platform const&) = delete;
        Platform& operator=(Platform&&) noexcept;
        ~Platform() noexcept;

        [[nodiscard]] uint64_t getSize() const;

        AppContext& ctx() &
        {
            return m_ctx;
        }

    private:
        AppContext m_ctx{toUnderlying(EPageSize::e1GB), toUnderlying(EPageSize::e1GB), {8192, 4096, 2048, 1024}};
    };
} // namespace dmt
