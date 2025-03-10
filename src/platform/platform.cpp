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

    bool AppContext::enabled(ELogLevel level) const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.enabled(level);
    }

    bool AppContext::traceEnabled() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.enabled(ELogLevel::TRACE);
    }

    bool AppContext::logEnabled() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.enabled(ELogLevel::LOG);
    }

    bool AppContext::warnEnabled() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.enabled(ELogLevel::WARNING);
    }

    bool AppContext::errorEnabled() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.enabled(ELogLevel::ERR);
    }

    void AppContext::dbgTraceStackTrace()
    {
        assert(m_pimpl);
        m_pimpl->mctx.pctx.dbgTraceStackTrace();
    }

    void AppContext::dbgErrorStackTrace()
    {
        assert(m_pimpl);
        m_pimpl->mctx.pctx.dbgErrorStackTrace();
    }

    size_t AppContext::maxLogArgBytes() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.maxLogArgBytes();
    }

    uint64_t AppContext::millisFromStart() const
    {
        assert(m_pimpl);
        return m_pimpl->mctx.pctx.millisFromStart();
    }

    void AppContext::addJob(Job const& job, EJobLayer layer)
    {
        assert(m_pimpl);
        m_pimpl->threadPool.addJob(m_pimpl->mctx, job, layer);
    }

    void AppContext::kickJobs()
    {
        assert(m_pimpl);
        m_pimpl->threadPool.kickJobs();
    }

    void AppContext::pauseJobs()
    {
        assert(m_pimpl);
        m_pimpl->threadPool.pauseJobs();
    }

    bool AppContext::otherLayerActive(EJobLayer& layer) const
    {
        assert(m_pimpl);
        return m_pimpl->threadPool.otherLayerActive(layer);
    }

    void* AppContext::stackAllocate(size_t size, size_t alignment, EMemoryTag tag, sid_t sid)
    {
        assert(m_pimpl);
        return m_pimpl->mctx.stackAllocate(size, alignment, tag, sid);
    }

    void AppContext::stackReset()
    {
        assert(m_pimpl);
        return m_pimpl->mctx.stackReset();
    }

    TaggedPointer AppContext::poolAllocateBlocks(uint32_t numBlocks, EBlockSize blockSize, EMemoryTag tag, sid_t sid)
    {
        assert(m_pimpl);
        return m_pimpl->mctx.poolAllocateBlocks(numBlocks, blockSize, tag, sid);
    }

    void AppContext::poolFreeBlocks(uint32_t numBlocks, TaggedPointer ptr)
    {
        assert(m_pimpl);
        return m_pimpl->mctx.poolFreeBlocks(numBlocks, ptr);
    }

    PageAllocationsTracker& AppContext::tracker()
    {
        assert(m_pimpl);
        return m_pimpl->mctx.tracker;
    }

    sid_t AppContext::intern(std::string_view str)
    {
        assert(m_pimpl);
        return m_pimpl->mctx.strTable.intern(str);
    }

    std::string_view AppContext::lookup(sid_t sid)
    {
        assert(m_pimpl);
        return m_pimpl->mctx.strTable.lookup(sid);
    }

    MemoryContext& AppContext::mctx()
    {
        assert(m_pimpl);
        return m_pimpl->mctx;
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

    void AppContext::log(std::string_view const& str, std::source_location const& loc)
    {
        write(ELogLevel::LOG, str, loc);
    }

    void AppContext::log(std::string_view const& str, std::initializer_list<StrBuf> const& list, std::source_location const& loc)
    {
        write(ELogLevel::LOG, str, list, loc);
    }

    void AppContext::warn(std::string_view const& str, std::source_location const& loc)
    {
        write(ELogLevel::WARNING, str, loc);
    }

    void AppContext::warn(std::string_view const& str, std::initializer_list<StrBuf> const& list, std::source_location const& loc)
    {
        write(ELogLevel::WARNING, str, list, loc);
    }

    void AppContext::error(std::string_view const& str, std::source_location const& loc)
    {
        write(ELogLevel::ERR, str, loc);
    }

    void AppContext::error(std::string_view const&              str,
                           std::initializer_list<StrBuf> const& list,
                           std::source_location const&          loc)
    {
        write(ELogLevel::ERR, str, list, loc);
    }

    void AppContext::trace(std::string_view const& str, std::source_location const& loc)
    {
        write(ELogLevel::TRACE, str, loc);
    }

    void AppContext::trace(std::string_view const&              str,
                           std::initializer_list<StrBuf> const& list,
                           std::source_location const&          loc)
    {
        write(ELogLevel::TRACE, str, list, loc);
    }

} // namespace dmt

namespace dmt::ctx {
    using namespace dmt;
    void init(AppContext& ctx)
    {
        uint64_t        pid = os::processId();
        std::lock_guard lk{detail::g_slk};
        auto            it = detail::g_ctxMap.find(pid);
        if (it == detail::g_ctxMap.end())
        {
            auto [it2, wasInserted] = detail::g_ctxMap.try_emplace(pid);
            assert(wasInserted);
            it = it2;
        }
        detail::CtxCtrlBlock& ctrl = it->second;
        std::lock_guard       lk2{ctrl.slk};
        if (ctrl.ctx != 0)
        {
            auto* curr = std::bit_cast<AppContext*>(&ctrl.ctx);
            std::destroy_at(curr);
            ctrl.ctx = 0;
        }
        // trygger copy constructor
        auto* ptr = std::bit_cast<AppContext*>(&ctrl.ctx);
        std::construct_at(ptr, ctx);
    }

    AppContext* acquireCurrent()
    {
        uint64_t pid = os::processId();
        auto     it  = detail::g_ctxMap.find(pid);
        if (it == detail::g_ctxMap.end())
        {
            assert(false);
            return nullptr;
        }
        detail::CtxCtrlBlock& ctrl = it->second;

        ctrl.slk.lock_shared();

        auto* ptr = std::bit_cast<AppContext*>(&ctrl.ctx);
        return ptr;
    }

    void releaseCurrent()
    {
        uint64_t pid = os::processId();
        auto     it  = detail::g_ctxMap.find(pid);
        if (it == detail::g_ctxMap.end())
        {
            assert(false);
        }
        else
        {
            detail::CtxCtrlBlock& ctrl = it->second;
            ctrl.slk.unlock_shared();
        }
    }

    void unregister()
    {
        uint64_t pid = os::processId();
        auto     it  = detail::g_ctxMap.find(pid);
        if (it == detail::g_ctxMap.end())
        {
            assert(false);
            return;
        }
        detail::CtxCtrlBlock& ctrl = it->second;
        std::unique_lock      lk0{ctrl.slk, std::defer_lock};
        std::unique_lock      lk{detail::g_slk, std::defer_lock};
        std::lock(lk0, lk);
        auto* curr = std::bit_cast<AppContext*>(&ctrl);
        std::destroy_at(curr);
        detail::g_ctxMap.erase(it);
    }
} // namespace dmt::ctx