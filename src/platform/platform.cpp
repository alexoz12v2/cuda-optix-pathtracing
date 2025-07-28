#include "platform.h"

#include <cstdint>
#include <cstdlib>
#include <cassert>

namespace dmt {
    [[nodiscard]] bool cudaDriverCall(NvcudaLibraryFunctions const* cudaApi, CUresult result)
    {
        if (result == ::CUDA_SUCCESS)
            return true;

        Context     ctx;
        char const* errorStr = nullptr;
        cudaApi->cuGetErrorString(result, &errorStr);

        if (errorStr != nullptr)
            ctx.error("CUDA Driver Error: {}", std::make_tuple(std::string_view(errorStr)));
        else
            ctx.error("CUDA Driver Unrecognized Error (did you forget to call cuInit?)", {});

        return false;
    }

    void fixCUDADriverSymbols(NvcudaLibraryFunctions* cudaApi)
    {
#if defined(DMT_OS_WINDOWS)
        // 1. apparently, as of CUDA 12.6, driver version ~540, cuCtxCreate_v4 is the latest creation
        // function, but we have a destroy up to v2
        cudaApi->cuCtxCreate = reinterpret_cast<decltype(cudaApi->cuCtxCreate)>(
            dmt::os::lib::getFunc(cudaApi->m_library, "cuCtxCreate_v2"));
        assert(cudaApi->cuCtxCreate);
        cudaApi->cuCtxDestroy = reinterpret_cast<decltype(cudaApi->cuCtxDestroy)>(
            dmt::os::lib::getFunc(cudaApi->m_library, "cuCtxDestroy_v2"));
        assert(cudaApi->cuCtxDestroy);
#endif
    }

    static std::recursive_mutex s_allocMutex;

    void Ctx::init(bool destroyIfExising, std::pmr::memory_resource* resource)
    {
        {
            std::scoped_lock lock{s_allocMutex};
            if (ctx::cs)
            {
                if (destroyIfExising)
                    destroy();
                else
                    return;
            }

            assert(!ctx::cs);
            m_resource = resource;
            ctx::cs    = reinterpret_cast<ctx::Contexts*>(resource->allocate(sizeof(ctx::Contexts)));
            std::construct_at(ctx::cs);
        }

        int32_t idx = -1;
        ctx::cs->addContext(false, &idx);
        ctx::cs->setActive(idx);
        Context ctx;
        ctx.impl()->addHandler([](dmt::LogHandler& _out) { dmt::createConsoleHandler(_out); });
    }

    void Ctx::destroy()
    {
        std::scoped_lock lock{s_allocMutex};
        std::destroy_at(ctx::cs);
        m_resource->deallocate(ctx::cs, sizeof(ctx::Contexts));
        m_resource = nullptr;
    }
} // namespace dmt
