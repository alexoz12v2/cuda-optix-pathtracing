#include "raygen-utils.h"

namespace dmt {
    GpuSamplerHandle uploadCpuDistrib(PiecewiseConstant2D const& cpuDistrib,
                                      filtering::Mitchell const& cpuFilter,
                                      CUDADriverLibrary const*   cudaApi)
    {
        GpuSamplerHandle handle;

        uint32_t const Nx = static_cast<uint32_t>(cpuDistrib.resolution().x);
        uint32_t const Ny = static_cast<uint32_t>(cpuDistrib.resolution().y);

        // Flatten conditional CDF (Nx+1 per row)
        std::vector<float> conditionalFlat;
        conditionalFlat.reserve(Ny * (Nx + 1));
        for (int y = 0; y < Ny; ++y)
        {
            auto const& rowCdf = cpuDistrib.conditionalCdfRow(y);
            conditionalFlat.insert(conditionalFlat.end(), rowCdf.begin(), rowCdf.end());
        }

        // Flatten marginal CDF
        auto const& marginalCdf = cpuDistrib.marginalCdf();

        // Allocate GPU memory
        cudaDriverCall(cudaApi,
                       cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&handle.dConditionalCdf),
                                           conditionalFlat.size() * sizeof(float)));
        cudaDriverCall(cudaApi,
                       cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&handle.dMarginalCdf),
                                           marginalCdf.size() * sizeof(float)));

        // Copy to GPU
        cudaDriverCall(cudaApi,
                       cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(handle.dConditionalCdf),
                                             conditionalFlat.data(),
                                             conditionalFlat.size() * sizeof(float)));
        cudaDriverCall(cudaApi,
                       cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(handle.dMarginalCdf),
                                             marginalCdf.data(),
                                             marginalCdf.size() * sizeof(float)));

        // Fill GPU structs
        handle.cudaApi                = cudaApi;
        handle.sampler.filter.radiusX = cpuFilter.radius().x;
        handle.sampler.filter.radiusY = cpuFilter.radius().y;
        handle.sampler.filter.B       = cpuFilter.b();
        handle.sampler.filter.C       = cpuFilter.c();

        handle.sampler.distrib.conditionalCdf = handle.dConditionalCdf;
        handle.sampler.distrib.marginalCdf    = handle.dMarginalCdf;
        handle.sampler.distrib.Nx             = Nx;
        handle.sampler.distrib.Ny             = Ny;
        handle.sampler.distrib.pMin           = {cpuDistrib.domain().pMin.x, cpuDistrib.domain().pMin.y};
        handle.sampler.distrib.pMax           = {cpuDistrib.domain().pMax.x, cpuDistrib.domain().pMax.y};
        handle.sampler.distrib.integral       = cpuDistrib.integral();

        return handle;
    }
} // namespace dmt