#pragma once

#include "cudautils/cudautils-macro.h"
#include "cudautils/cudautils.h"

// Maximum number of Halton dimensions (primes)
DMT_GPU __constant__ int d_primes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

// Each d_perms[i] points to a device array of uint16_t of length (nDigits * base)
DMT_GPU __constant__ uint16_t* d_perms[16];

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

    DMT_GPU dmt::float2 get2D() { return dmt::make_float2(get1D(), get1D()); }

    __device__ void nextSample()
    {
        haltonIndex++;
        dimension = 0;
    }
};