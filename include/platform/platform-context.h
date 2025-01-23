#pragma once

#include "dmtmacros.h"
#include "platform/platform-macros.h"
#include "platform/platform-utils.h"
#include "platform/platform-logging.h"

namespace dmt {
    // TODO move to logging
    // Will be valid only during the call to the handlers
    struct LogRecord
    {
        char8_t*       data;
        uint32_t       len;
        uint32_t       numBytes;
        dmt::ELogLevel level;
    };

    struct ContextImpl;
    class DMT_PLATFORM_API Context
    {
    public:
        Context();
        Context(Context const& _that);
        Context(Context&& _that) noexcept;
        Context& operator=(Context const& _that);
        Context& operator=(Context&& _that) noexcept;

    public:
        // Logging ----------------------------------------------------------------------------------------------------
        template <std::convertible_to<StrBuf>... Ts>
        void log()

            private : ContextImpl* m_pimpl;
    };
} // namespace dmt