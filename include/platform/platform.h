#pragma once

#include "dmtmacros.h"

#include <source_location>
#include <string_view>

#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-logging.h>
#include <platform/platform-memory.h>
#include <platform/platform-utils.h>
#else
import <platform/platform-logging.h>;
import <platform/platform-memory.h>;
import <platform/platform-utils.h>;
#endif

DMT_MODULE_EXPORT dmt {
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

        PlatformContext& ctx() &
        {
            return m_ctx;
        }

    private:
        static inline void doWrite(void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc)
        {
            Platform& self = *reinterpret_cast<Platform*>(data);
            self.m_logger.write(level, str, loc);
        }

        static inline void doWriteArgs(void*                                data,
                                       ELogLevel                            level,
                                       std::string_view const&              str,
                                       std::initializer_list<StrBuf> const& list,
                                       std::source_location const&          loc)
        {
            Platform& self = *reinterpret_cast<Platform*>(data);
            self.m_logger.write(level, str, list, loc);
        }

        static inline bool doCheckLevel(void* data, ELogLevel level)
        {
            Platform& self = *reinterpret_cast<Platform*>(data);
            return self.m_logger.enabled(level);
        }

        static inline void doChangeLevel(void* data, ELogLevel level)
        {
            reinterpret_cast<Platform*>(data)->m_logger.setLevel(level);
        }

        PlatformContext::InlineTable inlineTable() const
        {
            PlatformContext::InlineTable ret;
            ret.write      = doWrite;
            ret.writeArgs  = doWriteArgs;
            ret.checkLevel = doCheckLevel;
            return ret;
        }

        // Threadpool m_threadpool
        // Display m_display
        // ...
        ConsoleLogger          m_logger = ConsoleLogger::create();
        PlatformContext::Table m_ctxTable{.changeLevel = doChangeLevel};
        PlatformContext        m_ctx{this, &m_ctxTable, inlineTable()};
    };
} // namespace dmt
