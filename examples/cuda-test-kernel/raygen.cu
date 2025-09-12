#define DMT_CUDAUTILS_IMPLEMENTATION
#include "cudautils.h"
#include "cuda-queue.h"
#include "cudautils/cudautils-filter.h"

#define DMT_RAYGEN_WIDTH 1024

namespace dmt {

    struct RaygenPayload
    {
        // ox, oy, oz
        // dx, dy, dz
        // sample weight
        ManagedMultiQueue<float, float, float, float, float, float, float>* mmq;
    };

    struct RaygenParams
    {
        RaygenPayload rayPayload;
        int32_t       px;
        int32_t       py;
        int32_t       sampleIndex;
    };
} // namespace dmt

//REVIEW
#if 0
struct GpuSamplerHandle {
    FilterSamplerGPU sampler;
    float* dConditionalCdf = nullptr;
    float* dMarginalCdf    = nullptr;

    ~GpuSamplerHandle() {
        if (dConditionalCdf) cudaFree(dConditionalCdf);
        if (dMarginalCdf) cudaFree(dMarginalCdf);
    }
};

GpuSamplerHandle uploadCpuDistrib(const PiecewiseConstant2D& cpuDistrib,
                                  const Mitchell& cpuFilter) 
{
    GpuSamplerHandle handle;

    int Nx = cpuDistrib.resolution().x;
    int Ny = cpuDistrib.resolution().y;

    // Flatten conditional CDF (Nx+1 per row)
    std::vector<float> conditionalFlat;
    conditionalFlat.reserve(Ny * (Nx + 1));
    for (int y = 0; y < Ny; ++y) {
        auto const& rowCdf = cpuDistrib.m_pConditionalV[y].CDF();
        conditionalFlat.insert(conditionalFlat.end(), rowCdf.begin(), rowCdf.end());
    }

    // Flatten marginal CDF
    auto const& marginalCdf = cpuDistrib.m_pMarginalV.CDF();

    // Allocate GPU memory
    cudaMalloc(&handle.dConditionalCdf, conditionalFlat.size() * sizeof(float));
    cudaMalloc(&handle.dMarginalCdf, marginalCdf.size() * sizeof(float));

    // Copy to GPU
    cudaMemcpy(handle.dConditionalCdf, conditionalFlat.data(),
               conditionalFlat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(handle.dMarginalCdf, marginalCdf.data(),
               marginalCdf.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Fill GPU structs
    handle.sampler.filter.radiusX = cpuFilter.radius().x;
    handle.sampler.filter.radiusY = cpuFilter.radius().y;
    handle.sampler.filter.B       = cpuFilter.b();
    handle.sampler.filter.C       = cpuFilter.c();

    handle.sampler.distrib.conditionalCdf = handle.dConditionalCdf;
    handle.sampler.distrib.marginalCdf    = handle.dMarginalCdf;
    handle.sampler.distrib.Nx             = Nx;
    handle.sampler.distrib.Ny             = Ny;
    handle.sampler.distrib.pMin           = make_float2(cpuDistrib.domain().pMin.x, cpuDistrib.domain().pMin.y);
    handle.sampler.distrib.pMax           = make_float2(cpuDistrib.domain().pMax.x, cpuDistrib.domain().pMax.y);
    handle.sampler.distrib.integral       = cpuDistrib.integral();

    return handle;
}
#endif

extern "C"
{
    static __constant__ dmt::FilterSamplerGPU d_filter; // to be filled once by host
    static __constant__ dmt::Transform d_cameraFromRaster;
    static __constant__ dmt::Transform d_renderFromCamera;
}

namespace dmt {
    static __device__ Ray generateRayGPU(float2 p)
    {
        Point3f const pxImage{{p.x, p.y, 0}};
        Point3f const pCamera{{d_cameraFromRaster(pxImage)}};
        // TODO add lens?
        // time should use lerp(cs.time, shutterOpen, shutterClose)
        Ray const ray{Point3f{{0, 0, 0}}, normalize(pCamera), cs.time};
        // TODO handle tMax better
        float tMax = 1e5f;
        return d_renderFromCamera(ray, &tMax);
    }
} // namespace dmt

extern "C" __global__ void kraygen(RaygenParams params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = 0x12345678u ^ (params.px * 9781u + params.py * 6271u + params.sampleIndex * 13007u + tid ^ 0x4839'2482);
    dmt::DeviceSampler sampler{params.px, params.py, params.sampleIndex, seed};

    float sx  = params.px + 0.5f;
    float sy  = params.py + 0.5f;
    float pdf = 0;
    sampler.get2D(sx, sy);

    float2 sample = d_filter.sample(make_float2(sx, sy), &pdf);
    Ray    ray    = dmt::generateRayGPU(sample);
    // enqueue ray, pdf
    params.rayPayload.mmq->pushDevice(ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z, pdf);

    
}