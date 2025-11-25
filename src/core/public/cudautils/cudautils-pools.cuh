#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_POOLS_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_POOLS_CUH

#include "cudautils/cudautils-macro.cuh"

// std library
#include <cstdint>

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

}; // namespace dmt
#endif // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_POOLS_CUH
