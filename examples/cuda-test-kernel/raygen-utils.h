#pragma once

#include "core/core-math.h"
#include "core/core-render.h"
#include "cuda-wrappers-utils.h"
#include "cuda-wrappers/cuda-wrappers-cuda-driver.h"
#include "cudautils/cudautils-filter.h"

namespace dmt {
    struct GpuSamplerHandle
    {
        gpu::FilterSamplerGPU    sampler{};
        float*                   dConditionalCdf = nullptr;
        float*                   dMarginalCdf    = nullptr;
        CUDADriverLibrary const* cudaApi         = nullptr;

        ~GpuSamplerHandle()
        {
            if (dConditionalCdf)
                cudaDriverCall(cudaApi, cudaApi->cuMemFree(reinterpret_cast<CUdeviceptr>(dConditionalCdf)));
            if (dMarginalCdf)
                cudaDriverCall(cudaApi, cudaApi->cuMemFree(reinterpret_cast<CUdeviceptr>(dMarginalCdf)));
        }
    };

    GpuSamplerHandle uploadCpuDistrib(PiecewiseConstant2D const& cpuDistrib,
                                      filtering::Mitchell const& cpuFilter,
                                      CUDADriverLibrary const*   cudaApi);
} // namespace dmt