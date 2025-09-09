#define DMT_CUDAUTILS_IMPLEMENTATION
#include "cudautils.h"

struct RaygenParams
{
    int32_t px;
    int32_t py;
    int32_t sampleIndex;
};


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
}