#ifndef DMT_PLATFORM_PUBLIC_PLATFORM_CONTEXT_H
#define DMT_PLATFORM_PUBLIC_PLATFORM_CONTEXT_H

#include "dmtmacros.h"
#include "platform-macros.h"
#include "platform-utils.h"
#include "platform-logging.h"

namespace dmt {
    /**
     * @warning `Context` and `ContextImpl` are *not* thread safe classes. Instead, the services provided by the context,
     * namely, *logging*, *memory allocation and tracking*, *threadpool* have their services synchronized
     */
    // total size should be exactly 4KB in all platforms
    // should be allocated with `cudaMallocManaged` to make CUDA Path work properly
    struct DMT_PLATFORM_API ContextImpl
    {
    public:
        ContextImpl();
        ~ContextImpl();

        /**
         * @warning Thread unsafe, should be done when the process has a single thread owning the ContextImpl
         */
        template <typename F>
            requires std::is_invocable_v<F, LogHandler&>
        inline void addHandler(F&& f)
        {
            if (common.numHandlers >= maxHandlers)
                return;
            auto& ref = common.handlers[common.numHandlers++] = {};
            f(ref);
        }

        bool handlerEnabled(uint32_t i) const;

        bool anyHandlerEnabledFor(ELogLevel _level) const;

        bool logFilterPassesFor(uint32_t i, LogRecord const& record);

        void logCallbackFor(uint32_t i, LogRecord const& record);

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
            uint32_t lastLogTid  = 0;
            SpinLock mtx;

            // 1 byte aligned
            char logBuffer[logBufferNumBytes]{};
            char argLogBuffer[argLogBufferNumBytes]{};
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
        class DMT_PLATFORM_API Contexts
        {
        public:
            Contexts()                               = default;
            Contexts(Contexts const&)                = delete;
            Contexts(Contexts&&) noexcept            = delete;
            Contexts& operator=(Contexts const&)     = delete;
            Contexts& operator=(Contexts&&) noexcept = delete;
            ~Contexts();

        public:
            ECtxReturn addContext(bool managed = false, int32_t* outIdx = nullptr);

            bool setActive(int32_t index);

            ContextImpl* acquireActive();

            void releaseActive(ContextImpl** pCtx);

        private:
            void waitActiveUnused();

        private:
            struct DMT_PLATFORM_API Wrapper
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

        /**
         * Global variable which stores all contexts in use. the correct usage should be pass the context as parameter
         * everywhere and use a default as the first active context
         */
        extern DMT_PLATFORM_API Contexts* cs;
    } // namespace ctx

    /**
     * It's meant to be reacquired, so no copy control
     * @warning all `` methods must be implemented as `inline` here cause this is not a CUDA translation unit
     */
    class DMT_PLATFORM_API Context
    {
    public:
        Context();
        Context(Context const&)                = delete;
        Context(Context&&) noexcept            = delete;
        Context& operator=(Context const&)     = delete;
        Context& operator=(Context&&) noexcept = delete;
        ~Context();

        ContextImpl* impl();

    public:
        // TODO move inline stuff away
        inline bool isValid() const { return m_pimpl; }

        inline bool isLogEnabled() const { return m_pimpl && m_pimpl->anyHandlerEnabledFor(ELogLevel::LOG); }
        inline bool isWarnEnabled() const { return m_pimpl && m_pimpl->anyHandlerEnabledFor(ELogLevel::WARNING); }
        inline bool isTraceEnabled() const { return m_pimpl && m_pimpl->anyHandlerEnabledFor(ELogLevel::TRACE); }
        inline bool isErrorEnabled() const { return m_pimpl && m_pimpl->anyHandlerEnabledFor(ELogLevel::ERR); }

        // Logging ----------------------------------------------------------------------------------------------------
        template <typename... Ts>
        void log(FormatString<>              _fmt,
                 std::tuple<Ts...> const&    _params,
                 LogLocation const&          _pysLoc = getPhysicalLocation(),
                 std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::LOG, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        void trace(FormatString<>              _fmt,
                   std::tuple<Ts...> const&    _params,
                   LogLocation const&          _pysLoc = getPhysicalLocation(),
                   std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::TRACE, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        void warn(FormatString<>              _fmt,
                  std::tuple<Ts...> const&    _params,
                  LogLocation const&          _pysLoc = getPhysicalLocation(),
                  std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::WARNING, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        void error(FormatString<>              _fmt,
                   std::tuple<Ts...> const&    _params,
                   LogLocation const&          _pysLoc = getPhysicalLocation(),
                   std::source_location const& loc     = std::source_location::current())
        {
            write(_fmt, ELogLevel::ERR, _params, _pysLoc, loc);
        }

        template <typename... Ts>
        void write(FormatString<>              _fmt,
                   ELogLevel                   _level,
                   std::tuple<Ts...> const&    _params,
                   LogLocation const&          _pysLoc,
                   std::source_location const& loc)
        {
            std::lock_guard lk{m_pimpl->common.mtx};
            if (!m_pimpl->anyHandlerEnabledFor(_level))
                return;

            uint32_t const tid = os::threadId();
            if (m_pimpl->common.lastLogTid != tid)
            {
                flush();
                m_pimpl->common.lastLogTid = tid;
            }

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

    // TODO Define logging macros
} // namespace dmt
#endif // DMT_PLATFORM_PUBLIC_PLATFORM_CONTEXT_H
