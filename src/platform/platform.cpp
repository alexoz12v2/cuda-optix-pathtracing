#define DMT_INTERFACE_AS_HEADER
#undef DMT_NEEDS_MODULE
#include "platform.h"

#include <utility>

#include <cstdint>
#include <cstdlib>

namespace dmt {

    struct AppContextImpl
    {
        AppContextImpl(uint32_t                                   pageTrackCapacity,
                       uint32_t                                   allocTrackCapacity,
                       std::array<uint32_t, numBlockSizes> const& numBlocksPerPool) :
        mctx{pageTrackCapacity, allocTrackCapacity, numBlocksPerPool},
        threadPool{mctx},
        refCount(1)
        {
        }

        MemoryContext        mctx;
        ThreadPoolV2         threadPool;
        std::atomic<int32_t> refCount; // should only manage copy control of context, not synchronize member functions
    };

    AppContext::AppContext(uint32_t                                   pageTrackCapacity,
                           uint32_t                                   allocTrackCapacity,
                           std::array<uint32_t, numBlockSizes> const& numBlocksPerPool) :
    m_pimpl{new AppContextImpl(pageTrackCapacity, allocTrackCapacity, numBlocksPerPool)}
    {
    }

    AppContext::AppContext(AppContext const& other) : m_pimpl(other.m_pimpl)
    {
        assert(m_pimpl);
        m_pimpl->refCount.fetch_add(1, std::memory_order_release);
    }

    AppContext::AppContext(AppContext&& other) noexcept : m_pimpl(std::exchange(other.m_pimpl, nullptr)) {}

    AppContext& AppContext::operator=(AppContext const& that)
    {
        assert(m_pimpl);
        if (this != &that)
        {
            cleanup();
            m_pimpl = that.m_pimpl;
            m_pimpl->refCount.fetch_add(1, std::memory_order_release);
        }
        return *this;
    }

    AppContext& AppContext::operator=(AppContext&& that) noexcept
    {
        assert(m_pimpl);
        if (this != &that)
        {
            m_pimpl = std::exchange(that.m_pimpl, nullptr);
        }
        return *this;
    }

    void AppContext::write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
    {
        assert(m_pimpl);
        m_pimpl->mctx.pctx.write(level, str, loc);
    }

    void AppContext::write(ELogLevel                            level,
                           std::string_view const&              str,
                           std::initializer_list<StrBuf> const& list,
                           std::source_location const&          loc)
    {
        assert(m_pimpl);
        m_pimpl->mctx.pctx.write(level, str, list, loc);
    }

    size_t AppContext::maxLogArgBytes() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.maxLogArgBytes();
    }

    void AppContext::addJob(Job const& job, EJobLayer layer)
    {
        assert(m_pimpl);
        m_pimpl->threadPool.addJob(m_pimpl->mctx, job, layer);
    }

    static void cleanupContext(AppContextImpl* pctx)
    {
        if (pctx->refCount.fetch_sub(1, std::memory_order_acquire) <= 1)
        {
            pctx->threadPool.cleanup(pctx->mctx);
            pctx->mctx.cleanup();
            delete pctx;
        }
    }

    void AppContext::cleanup() noexcept
    {
        assert(m_pimpl);
        cleanupContext(m_pimpl);
    }

    AppContext::~AppContext() noexcept { cleanup(); }

} // namespace dmt

namespace dmt::ctx {
    using namespace dmt;
    void init(AppContext& ctx)
    {
        std::lock_guard lk{detail::g_slk};
        if (detail::g_currentContext != 0)
        {
            auto* curr = std::bit_cast<AppContext*>(&detail::g_currentContext);
            std::destroy_at(curr);
            detail::g_currentContext = 0;
        }
        // trygger copy constructor
        auto* ptr = std::bit_cast<AppContext*>(&detail::g_currentContext);
        std::construct_at(ptr, ctx);
    }

    AppContext* acquireCurrent()
    {
        assert(detail::g_currentContext);
        detail::g_slk.lock_shared();
        auto* ptr = std::bit_cast<AppContext*>(&detail::g_currentContext);
        return ptr;
    }

    DMT_PLATFORM_MIXED_API void releaseCurrent()
    {
        assert(detail::g_currentContext);
        detail::g_slk.unlock_shared();
    }

    void unregister()
    {
        assert(detail::g_currentContext);
        std::lock_guard lk{detail::g_slk};
        auto*           curr = std::bit_cast<AppContext*>(&detail::g_currentContext);
        std::destroy_at(curr);
        detail::g_currentContext = 0;
    }
} // namespace dmt::ctx