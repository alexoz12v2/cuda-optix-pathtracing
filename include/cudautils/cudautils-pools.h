#pragma once
#include "cudautils/cudautils-macro.h"
#include <algorithm>
#include <cstdint>
#include <memory>

namespace dmt {

    struct FilmSOA
    {
        float *r, *g, *b, *w;
        int    width, height;
    };

    struct HitPool
    {
        float *  t, *u, *v;       // bary, t
        int32_t *inst, *tri;      // hit ids
        float *  ngx, *ngy, *ngz; // geometric normal (object or world)
        int      capacity;
    };

    struct RayPool
    {
        float *   ox, *oy, *oz; //origin
        float *   dx, *dy, *dz; //direction
        float *   tmin, *tmax;
        float *   beta_r, *beta_g, *beta_b;
        float*    bsdfPdf; // last bsdf pdf
        int*      sampleIdx;
        int32_t*  depth;                 // current depth
        int32_t * pixel_x, *pixel_y;     // pixel coords for film
        uint32_t *rngState0, *rngState1; // RNG state per ray (or counter)
        uint32_t  capacity;
        uint32_t  size; // device-side atomic count for enqueue/dequeue
    };

    void allocHitPool(CUDADriverLibrary* cudaApi, HitPool& H, uint32_t n)
    {
        H.capacity = n;
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.t), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.u), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.v), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.inst), sizeof(int32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.tri), sizeof(int32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.ngx), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.ngy), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&H.ngz), sizeof(float) * n);
    }
    void allocFilm(CUDADriverLibrary* cudaApi, FilmSOA& F, int W, int H)
    {
        F.width    = W;
        F.height   = H;
        uint32_t N = W * H;
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&F.r), sizeof(float) * N);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&F.g), sizeof(float) * N);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&F.b), sizeof(float) * N);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&F.w), sizeof(float) * N);
        cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(F.r), 0, sizeof(float) * N);
        cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(F.g), 0, sizeof(float) * N);
        cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(F.b), 0, sizeof(float) * N);
        cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(F.w), 0, sizeof(float) * N);
    }

    void allocRayPool(CUDADriverLibrary* cudaApi, RayPool& P, uint32_t n)
    {
        P.capacity = n;
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.ox), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.oy), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.oz), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.dx), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.dy), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.dz), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.tmin), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.tmax), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.beta_r), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.beta_g), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.beta_b), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.bsdfPdf), sizeof(float) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.depth), sizeof(int32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.pixel_y), sizeof(int32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.pixel_x), sizeof(int32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.sampleIdx), sizeof(uint32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.rngState0), sizeof(uint32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&P.rngState1), sizeof(uint32_t) * n);
    }
}; // namespace dmt
