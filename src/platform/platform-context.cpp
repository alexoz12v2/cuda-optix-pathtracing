#include "platform-context.h"

#include <thread>

namespace dmt {
    ContextImpl::ContextImpl() {}

    ContextImpl::~ContextImpl()
    {
        for (uint32_t i = 0; i < common.numHandlers; ++i)
            common.handlers[i].hostCleanup(common.handlers[i].hostDeallocate, common.handlers[i].data);
    }

    bool ContextImpl::handlerEnabled(uint32_t i) const { return common.handlers[i].minimumLevel < ELogLevel::NONE; }

    bool ContextImpl::anyHandlerEnabledFor(ELogLevel _level) const
    {
        assert(_level != ELogLevel::NONE);
        for (uint32_t i = 0; i < common.numHandlers; ++i)
        {
            if (common.handlers[i].minimumLevel <= _level)
                return true;
        }
        return false;
    }

    bool ContextImpl::logFilterPassesFor(uint32_t i, LogRecord const& record)
    {
        return common.handlers[i].hostFilter(common.handlers[i].data, record);
    }

    void ContextImpl::logCallbackFor(uint32_t i, LogRecord const& record)
    {
        common.handlers[i].hostCallback(common.handlers[i].data, record);
    }

    Context::Context() : m_pimpl(ctx::cs ? ctx::cs->acquireActive() : nullptr) {}

    Context::~Context() { ctx::cs->releaseActive(&m_pimpl); }

    ContextImpl* Context::impl() { return m_pimpl; }

    void Context::flush()
    {
        for (uint32_t i = 0; i < m_pimpl->common.numHandlers; ++i)
        {
            if (m_pimpl->handlerEnabled(i))
                m_pimpl->common.handlers[i].hostFlush(m_pimpl->common.handlers[i].data);
        }
    }

    // CONTEXTS -------------------------------------------------------------------------------------------------------
    namespace ctx {
        Contexts::~Contexts()
        {
            waitActiveUnused();
            std::lock_guard lk{lock};
            // destroy each context and deallocate its memory
            for (int32_t i = 0; i < count; ++i)
            {
                std::destroy_at(ctxs[i].pctx);
                //if (ctxs[i].gpu)
                //    cudaFree(ctxs[i].pctx);
                //else
                {
                    ::dmt::os::deallocate(ctxs[i].pctx, sizeof(ContextImpl), alignof(ContextImpl));
                }
            }
        }

        ECtxReturn Contexts::addContext(bool managed, int32_t* outIdx)
        {
            std::lock_guard lk{lock};
            //if (managed)
            //{
            //    cudaError_t err = cudaMallocManaged(&ctxs[count].pctx, sizeof(ContextImpl));
            //    if (err != ::cudaSuccess)
            //        ctxs[count].pctx = nullptr;
            //    else
            //        ctxs[count].gpu = true;
            //}

            if (!ctxs[count].pctx)
            {
                ctxs[count].pctx = reinterpret_cast<ContextImpl*>(
                    ::dmt::os::allocate(sizeof(ContextImpl), alignof(ContextImpl)));
                if (!ctxs[count].pctx)
                    return ECtxReturn::eMemoryError;
                ctxs[count].gpu = false;
            }

            ctxs[count].readCount = 0;
            std::construct_at(ctxs[count].pctx);
            if (outIdx)
                *outIdx = count;
            ++count;
            return ctxs[count - 1].gpu ? ECtxReturn::eCreatedOnManaged : ECtxReturn::eCreatedOnHost;
        }

        bool Contexts::setActive(int32_t index)
        {
            waitActiveUnused();
            std::lock_guard lk{lock};
            // wait on all read counts to be zero
            if (index < 0 || index >= count)
                return false;
            activeIndex = index;
            return true;
        }

        ContextImpl* Contexts::acquireActive()
        {
            ContextImpl* ret = nullptr;
            lock.lock_shared();
            {
                // increment `readCount`
                atomic::increment(&ctxs[activeIndex].readCount);
                ret = ctxs[activeIndex].pctx;
            }
            lock.unlock_shared();
            return ret;
        }

        void Contexts::releaseActive(ContextImpl** pCtx)
        {
            lock.lock_shared();
            {
                auto old = atomic::decrement(&ctxs[activeIndex].readCount);
                if (old <= 0)
                {
                    std::abort();
                }
                *pCtx = nullptr;
            }
            lock.unlock_shared();
        }

        void Contexts::waitActiveUnused()
        {
            bool ready = false;
            while (!ready)
            {
                ready = true;
                for (int32_t i = 0; i < count; ++i)
                {
                    if (std::atomic_ref(ctxs[i].readCount).load() > 0)
                    {
                        ready = false;
                        std::this_thread::yield();
                        break;
                    }
                }
            }
        }

        DMT_PLATFORM_API Contexts* cs = nullptr;
    } // namespace ctx
} // namespace dmt