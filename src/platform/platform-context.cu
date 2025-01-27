#include "platform-context.h"
#include <cuda_runtime.h>

namespace dmt {
    ContextImpl::ContextImpl() {}

    ContextImpl::~ContextImpl()
    {
        for (uint32_t i = 0; i < common.numHandlers; ++i)
            common.handlers[i].hostCleanup(common.handlers[i].hostDeallocate, common.handlers[i].data);
    }

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
        __host__ Contexts::~Contexts()
        {
            waitActiveUnused();
            std::lock_guard lk{lock};
            // destroy each context and deallocate its memory
            for (int32_t i = 0; i < count; ++i)
            {
                std::destroy_at(ctxs[i].pctx);
                if (ctxs[i].gpu)
                    cudaFree(ctxs[i].pctx);
                else
                    ::dmt::deallocate(ctxs[i].pctx, sizeof(ContextImpl), alignof(ContextImpl));
            }
        }

        __host__ ECtxReturn Contexts::addContext(int32_t* outIdx)
        {
            std::lock_guard lk{lock};
            cudaError_t err = cudaMallocManaged(&ctxs[count].pctx, sizeof(ContextImpl));
            if (err != ::cudaSuccess)
                ctxs[count].pctx = nullptr;
            else
                ctxs[count].gpu = true;

            if (!ctxs[count].pctx)
            {
                ctxs[count].pctx = reinterpret_cast<ContextImpl*>(
                    ::dmt::allocate(sizeof(ContextImpl), alignof(ContextImpl)));
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

        __host__ bool Contexts::setActive(int32_t index)
        {
            waitActiveUnused();
            std::lock_guard lk{lock};
            // wait on all read counts to be zero
            if (index < 0 || index >= count)
                return false;
            activeIndex = index;
            return true;
        }

        __host__ __device__ ContextImpl* Contexts::acquireActive()
        {
            ContextImpl* ret = nullptr;
            lock.lock_shared();
            {
#if defined(__CUDA_ARCH__)
                if (!ctxs[activeIndex].gpu)
                    asm("trap;"); // TODO better
#endif
                // increment `readCount`
                atomic::increment(&ctxs[activeIndex].readCount);
                ret = ctxs[activeIndex].pctx;
            }
            lock.unlock_shared();
            return ret;
        }

        __host__ __device__ void Contexts::releaseActive(ContextImpl** pCtx)
        {
            lock.lock_shared();
            {
#if defined(__CUDA_ARCH__)
                if (!ctxs[activeIndex].gpu)
                    asm("trap;"); // TODO better
#endif
                auto old = atomic::decrement(&ctxs[activeIndex].readCount);
                if (old <= 0)
                {
#if defined(__CUDA_ARCH__)
                    asm("trap;");
#else
                    std::abort();
#endif
                }
                *pCtx = nullptr;
            }
            lock.unlock_shared();
        }

        __host__ void Contexts::waitActiveUnused()
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

        __managed__ Contexts* cs;

        __host__ ECtxReturn addContext(bool managed, int32_t* outIdx)
        {
            if (!cs)
            {
                if (managed)
                {
                    cudaError_t err = cudaMallocManaged(&cs, sizeof(Contexts));
                    if (err != ::cudaSuccess)
                        std::abort();
                }
                else
                {
                    cs = new Contexts;
                    if (!cs)
                        std::abort();
                }
            }
            return cs->addContext(outIdx);
        }
    } // namespace ctx
} // namespace dmt