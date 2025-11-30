#ifndef __NVCC__
    #define __NVCC__
#endif

// ==== GLM ==== (hacks to make it work under JIT compilation)
#include "cuda.h"
#ifndef CUDA_VERSION
    #define CUDA_VERSION 11800
#endif
#define GLM_FORCE_CUDA
#define GLM_FORCE_PURE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// EIGEN not Made for JIT compilation
//// ==== Eigen ====
//#define EIGEN_DEVICE_FUNC __host__ __device__
//#define EIGEN_NO_DEBUG
//#define EIGEN_DISABLE_THREADS
//#define EIGEN_USE_GPU
//#include <Eigen/Core>

#define DMT_CUDAUTILS_IMPLEMENTATION
#include "cudautils/cudautils.h"
#include "cuda-queue.h"
#include "cudautils/cudautils-filter.h"

#define DMT_RAYGEN_WIDTH 1024

//REVIEW

extern "C"
{
    static __constant__ dmt::gpu::FilterSamplerGPU d_filter; // to be filled once by host
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
        Ray const ray{Point3f{{0, 0, 0}}, normalize(pCamera), /*time*/ 0.f};
        // TODO handle tMax better
        float tMax = 1e5f;
        return d_renderFromCamera(ray, &tMax);
    }
} // namespace dmt

extern "C" __global__ void kraygen(dmt::RaygenParams params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = 0x12345678u ^ (params.px * 9781u + params.py * 6271u + params.sampleIndex * 13007u + tid ^ 0x4839'2482);
    dmt::DeviceSampler sampler{params.px, params.py, static_cast<uint32_t>(params.sampleIndex), seed};

    float sx  = params.px + 0.5f;
    float sy  = params.py + 0.5f;
    float pdf = 0;
    sampler.get2D(sx, sy);

    float2   sample = d_filter.sample(make_float2(sx, sy), &pdf);
    dmt::Ray ray    = dmt::generateRayGPU(sample);
    // enqueue ray, pdf
    params.rayPayload.mmq->pushDevice(ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z, pdf);
}