#pragma once

#include "cudautils/cudautils-filter.h"
#include "cudautils/cudautils-macro.h"
#include "cudautils/cudautils-vecmath.h"

namespace dmt {

    struct GpuSamplerHandle
    {
        gpu::FilterSamplerGPU sampler;
        float*           dConditionalCdf = nullptr;
        float*           dMarginalCdf    = nullptr;

        //static inline void freeGpuSamplerHandle(CUDADriverLibrary const* cudaApi)
        //{
        //    if (dConditionalCdf)
        //        cudaFree(dConditionalCdf);
        //    if (dMarginalCdf)
        //        cudaFree(dMarginalCdf);
        //}
    };

#if 0
    // Maximum number of Halton dimensions (primes)
    DMT_GPU __constant__ int d_primes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    // Each d_perms[i] points to a device array of uint16_t of length (nDigits * base)
    DMT_GPU __constant__ uint16_t* d_perms[16];

    // TODO change. A warp will generate the same exact samples. This class is called by GPU code only (warp might have a non full mask)
    struct GpuHaltonOwenSampler
    {
        int64_t haltonIndex;
        int32_t dimension;

        DMT_GPU void startSample(int64_t sampleIdx)
        {
            haltonIndex = sampleIdx;
            dimension   = 0;
        }

        DMT_GPU float vanDerCorput(int64_t index, int base, uint16_t const* perm, int nDigits) const
        {
            float result   = 0.f;
            float f        = 1.f / base;
            int   digitIdx = 0;
            while (index > 0)
            {
                int digit = index % base;
                if (digitIdx < nDigits)
                    digit = perm[digitIdx * base + digit]; // Owen permutation
                result += digit * f;
                index /= base;
                f /= base;
                digitIdx++;
            }
            return result;
        }

        DMT_GPU float get1D()
        {
            int             dim     = dimension;
            int             base    = d_primes[dim];
            uint16_t const* perm    = d_perms[dim];
            int             nDigits = 64; // can adjust
            float           val     = vanDerCorput(haltonIndex, base, perm, nDigits);
            dimension++;
            return val;
        }

        DMT_GPU Vector2f get2D() { return {get1D(), get1D()}; }

        __device__ void nextSample()
        {
            haltonIndex++;
            dimension = 0;
        }
    };
#endif
#if defined(__CUDA_ARCH__)
    __device__ inline uint32_t wang_hash(uint32_t x)
    {
        x = (x ^ 61u) ^ (x >> 16);
        x *= 9u;
        x = x ^ (x >> 4);
        x *= 0x27d4eb2du;
        x = x ^ (x >> 15);
        return x;
    }

    // float in [0,1)
    inline __device__ float hash_to_float(uint32_t h)
    {
        // divide by 2^32
        return (float)h * (1.0f / 4294967296.0f);
    }

    // radical inverse in arbitrary base (works for small bases like 2,3)
    inline __device__ float radical_inverse_base(uint32_t base, uint64_t a)
    {
        float invBase = 1.0f / (float)base;
        float inv     = invBase;
        float result  = 0.0f;
        while (a)
        {
            uint32_t digit = (uint32_t)(a % base);
            result += digit * inv;
            a /= base;
            inv *= invBase;
        }
        return result;
    }

    // Cranley-Patterson rotation: add per-pixel offset (seeded) and wrap
    // Inputs are:
    //  - pixelHashSeed: a 32-bit hash derived from pixel coords and a seed
    //  - baseSample: the low-discrepancy sample value in [0,1)
    // returns: sample in [0,1)
    inline __device__ float cranley_patterson_rotate(float baseSample, uint32_t pixelHashSeed)
    {
        float offset = hash_to_float(pixelHashSeed);
        float v      = baseSample + offset;
        // fract
        return v - floorf(v);
    }

    // Example: sample 2D within pixel using Halton primes 2 and 3 with Cranley-Patterson rotation
    // pixelX, pixelY: pixel coordinates
    // sppIndex: sample index for this pixel (0..spp-1)
    // dimOffset: use to get different dims (0->use primes dims 0&1, 1->next dims if you chain)
    inline __device__ void samplePixel2D_halton_cp(
        int32_t  pixelX,
        int32_t  pixelY,
        uint32_t sppIndex,
        uint32_t dimOffset,
        uint32_t globalSeed,
        float&   out_u,
        float&   out_v)
    {
        // Use primes 2 and 3 (your original approach). If you want more dims,
        // use other primes or Sobol.
        uint32_t const primes[2] = {2u, 3u};

        // Compose a per-pixel seed/hash
        uint32_t pixHash = wang_hash((uint32_t)pixelX * 0x9e3779b9u + (uint32_t)pixelY);
        pixHash          = wang_hash(pixHash + globalSeed);

        // Build unique hash per-dimension (so offsets for x and y differ)
        uint32_t pixHashX = wang_hash(pixHash + 0x9137u + dimOffset);
        uint32_t pixHashY = wang_hash(pixHash + 0x5f39u + dimOffset);

        // compute radical inverses from the sample index (indexable, no state)
        uint64_t index = (uint64_t)sppIndex; // you can incorporate more offsets here
        float    rx    = radical_inverse_base(primes[0], index);
        float    ry    = radical_inverse_base(primes[1], index);

        // Apply Cranley-Patterson rotation to decorrelate pixels
        out_u = cranley_patterson_rotate(rx, pixHashX);
        out_v = cranley_patterson_rotate(ry, pixHashY);
    }

    struct DeviceSampler
    {
        int32_t  px, py;
        uint32_t sampleIndex;
        uint32_t seed;
        int32_t  dim;

        __device__ DeviceSampler(int32_t x, int32_t y, uint32_t s, uint32_t seed0) :
        px(x),
        py(y),
        sampleIndex(s),
        seed(seed0),
        dim(0)
        {
        }

        __device__ float get1D()
        {
            // pick a base or use sobol dim 'dim'
            uint32_t dimOff = (uint32_t)dim++;
            float    u;
            float    tmp;
            samplePixel2D_halton_cp(px, py, sampleIndex, dimOff, seed, u, tmp);
            return u;
        }

        __device__ void get2D(float& u, float& v)
        {
            uint32_t d = (uint32_t)(dim);
            samplePixel2D_halton_cp(px, py, sampleIndex, d / 2 /*or other mapping*/, seed, u, v);
            dim += 2;
        }
    };
#endif
} // namespace dmt