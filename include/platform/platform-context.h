#pragma once

#include "dmtmacros.h"
#include "platform/platform-macros.h"
#include "platform/platform-utils.h"
#include "platform/platform-logging.h"

namespace dmt {
    /**
     * @warning `Context` and `ContextImpl` are *not* thread safe classes. Instead, the services provided by the context,
     * namely, *logging*, *memory allocation and tracking*, *threadpool* have their services synchronized
     */
    // total size should be exactly 4KB in all platforms
    // should be allocated with `cudaMallocManaged` to make CUDA Path work properly
    struct ContextImpl
    {
    public:
        DMT_CPU ContextImpl();
        DMT_CPU ~ContextImpl();

        /**
         * @warning Thread unsafe, should be done when the process has a single thread owning the ContextImpl
         */
        template <typename F>
            requires std::is_invocable_v<F, LogHandler&>
        inline DMT_CPU void addHandler(F&& f)
        {
            if (common.numHandlers >= maxHandlers)
                return;
            auto& ref = common.handlers[common.numHandlers++] = {};
            f(ref);
        }

        inline DMT_CPU_GPU bool handlerEnabled(uint32_t i) const
        {
            return common.handlers[i].minimumLevel < ELogLevel::NONE;
        }

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
        struct Common
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

    // global context management functions
    namespace ctx {
        enum class ECtxReturn
        {
            eCreatedOnManaged,
            eCreatedOnHost,
            eMaxReached,
            eMemoryError,
            eGenericError
        };
        inline constexpr uint32_t maxNumCtxs = 4;
        class Contexts
        {
        public:
            Contexts()                               = default;
            Contexts(Contexts const&)                = delete;
            Contexts(Contexts&&) noexcept            = delete;
            Contexts& operator=(Contexts const&)     = delete;
            Contexts& operator=(Contexts&&) noexcept = delete;
            DMT_CPU ~Contexts();

        public:
            DMT_CPU ECtxReturn addContext(bool managed = false, int32_t* outIdx = nullptr);

            DMT_CPU bool setActive(int32_t index);

            DMT_CPU_GPU ContextImpl* acquireActive();

            DMT_CPU_GPU void releaseActive(ContextImpl** pCtx);

        private:
            DMT_CPU void waitActiveUnused();

        private:
            struct Wrapper
            {
                ContextImpl* pctx      = nullptr;
                int32_t      readCount = 0;
                bool         gpu       = false;
            };

        private:
            Wrapper         ctxs[maxNumCtxs];
            int32_t         count       = 0;
            int32_t         activeIndex = -1; // managed with atomic_ref
            CudaSharedMutex lock;
        };
        extern DMT_MANAGED Contexts* cs;

        /**
         * @warning thread unsafe
         * @warning memory leak is on purpose
         */
        DMT_CPU ECtxReturn addContext(bool managed = false, int32_t* outIdx = nullptr);
    } // namespace ctx

    /**
     * Buffer to facilitate GPU side logging
     * Requirements:
     * - call "LogBuffer" = activeMask + sizeof(LogRecord) + UTF-8 character buffer for the format string
     * - there should be N * warpSize LogBuffers allocated in managed memory
     * - when
     * @warning should be constructed in `__managed__` memory
     */
    class WarpBuffers
    {
    public:
        struct Buffer
        {
            LogRecord record[32];
            char      arg[1024][32];
            char      buf[1024][32];
            bool      used[32];
        };

    public:
        DMT_CPU WarpBuffers();
        WarpBuffers(WarpBuffers const&)                = delete;
        WarpBuffers(WarpBuffers&&) noexcept            = delete;
        WarpBuffers& operator=(WarpBuffers const&)     = delete;
        WarpBuffers& operator=(WarpBuffers&&) noexcept = delete;
        DMT_CPU ~WarpBuffers();

    public:

    private:

    private:
    };

    /**
     * It's meant to be reacquired, so no copy control
     * @warning all `DMT_CPU_GPU` methods must be implemented as `inline` here cause this is not a CUDA translation unit
     */
    class Context
    {
    public:
        DMT_CPU_GPU          Context() : m_pimpl(ctx::cs->acquireActive()) {}
        DMT_CPU_GPU          Context(Context const&)       = delete;
        DMT_CPU_GPU          Context(Context&&) noexcept   = delete;
        DMT_CPU_GPU Context& operator=(Context const&)     = delete;
        DMT_CPU_GPU Context& operator=(Context&&) noexcept = delete;
        DMT_CPU_GPU ~Context()
        { //
            ctx::cs->releaseActive(&m_pimpl);
        }

        inline ContextImpl* impl() { return m_pimpl; }

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
        DMT_CPU_GPU void trace(FormatString<>              _fmt,
                               std::tuple<Ts...> const&    _params,
                               LogLocation const&          _pysLoc = getPhysicalLocation(),
                               std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::TRACE, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        DMT_CPU_GPU void warn(FormatString<>              _fmt,
                              std::tuple<Ts...> const&    _params,
                              LogLocation const&          _pysLoc = getPhysicalLocation(),
                              std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::WARNING, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        DMT_CPU_GPU void error(FormatString<>              _fmt,
                               std::tuple<Ts...> const&    _params,
                               LogLocation const&          _pysLoc = getPhysicalLocation(),
                               std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::ERR, _params, _pysLoc, loc);
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
                // on the GPU, take a free buffer from common.gpuLogBuffers, if not nullptr,
                // create a record using those buffers, (each buffer contains 2 char buffers, a log record, and a bool to say whether it is being used)
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
        void flush();

    private:
        ContextImpl* m_pimpl;
    };
} // namespace dmt