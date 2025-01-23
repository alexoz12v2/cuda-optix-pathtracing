#pragma once

#include "dmtmacros.h"
#include "platform/platform-macros.h"
#include "platform/platform-utils.h"
#include "platform/platform-logging.h"

namespace dmt {
    // should reside in unified memory in CUDA path
    struct DMT_PLATFORM_API LogHandler
    {
        ELogLevel minimumLevel;
        void*     data;
        bool (*hostFilter)(void* _data, LogRecord const& record);
        void (*hostCallback)(void* _data, LogRecord const& record);
    };

    // total size should be exactly 4KB in all platforms
    // should be allocated with `cudaMallocManaged` to make CUDA Path work properly
    struct DMT_PLATFORM_API ContextImpl
    {
    public:
        DMT_CPU ContextImpl();
        DMT_CPU ~ContextImpl();

        DMT_CPU bool addHandler(LogHandler handler);

        inline DMT_CPU_GPU bool anyHandlerEnabledFor(ELogLevel _level) const
        {
            assert(_level != ELogLevel::NONE);
            for (uint32_t i = 0; i < common.numHandlers; ++i)
            {
                if (common.handlers[i].minimumLevel <= _level)
                    return true;
            }
            return false;
        }

        inline DMT_CPU bool logFilterPassesFor(uint32_t i, LogRecord const& record)
        {
            return common.handlers[i].hostFilter(common.handlers[i].data, record);
        }

        inline DMT_CPU void logCallbackFor(uint32_t i, LogRecord const& record)
        {
            common.handlers[i].hostCallback(common.handlers[i].data, record);
        }

    public:
        static constexpr uint32_t maxHandlers          = 4;
        static constexpr uint32_t logBufferNumBytes    = 2048;
        static constexpr uint32_t argLogBufferNumBytes = 1024;
        struct DMT_PLATFORM_API Common
        {
            // 8 byte aligned
            LogHandler handlers[maxHandlers]{};

            // 4 byte aligned
            uint32_t numHandlers = 0;

            // 1 byte aligned
            char8_t logBuffer[logBufferNumBytes]{};
            char8_t argLogBuffer[argLogBufferNumBytes]{};
        };

        Common common;
        // implicit padding to make next address 8 byte aligned
        // followed by platform specific data
        alignas(std::max_align_t) unsigned char platformSpecific[4096 - sizeof(Common)]{};
    };
    static_assert(std::is_standard_layout_v<ContextImpl>);

    class DMT_PLATFORM_API Context
    {
    public:
        DMT_CPU_GPU Context(ContextImpl* ptr) : m_pimpl(ptr) {}

    public:
        // Logging ----------------------------------------------------------------------------------------------------
        template <typename... Ts>
        DMT_CPU_GPU void log(FormatString<>              _fmt,
                             std::tuple<Ts...> const&    _params,
                             LogLocation const&          _pysLoc = getPhysicalLocation(),
                             std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::LOG, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        DMT_CPU_GPU void write(FormatString<>              _fmt,
                               ELogLevel                   _level,
                               std::tuple<Ts...> const&    _params,
                               LogLocation const&          _pysLoc,
                               std::source_location const& loc)
        {
            if (!m_pimpl->anyHandlerEnabledFor(_level))
                return;

#if defined(__CUDA_ARCH__)
                // TODO: Dump everything on a buffer
#else
            auto record = createRecord(_fmt,
                                       _level,
                                       m_pimpl->common.logBuffer,
                                       ContextImpl::logBufferNumBytes,
                                       m_pimpl->common.argLogBuffer,
                                       ContextImpl::argLogBufferNumBytes,
                                       _params,
                                       _pysLoc,
                                       loc);
            for (uint32_t i = 0; i < m_pimpl->common.numHandlers; ++i)
            {
                if (m_pimpl->common.handlers[i].minimumLevel <= _level && m_pimpl->logFilterPassesFor(i, record))
                {
                    m_pimpl->logCallbackFor(i, record);
                }
            }
#endif
        }

    private:
        ContextImpl* m_pimpl;
    };
} // namespace dmt