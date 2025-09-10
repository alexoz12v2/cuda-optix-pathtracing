#pragma once

#include "cudautils/cudautils-macro.h"

#if 0
struct DeviceFilter
{
    float* d_f;           // Ny * Nx
    float* d_condCdf;     // Ny * (Nx+1)
    float* d_marginalCdf; // Ny+1
    int    Nx, Ny;
    float  xMin, xMax, yMin, yMax;
    float  integral;
};

#if defined(__CUDA_ARCH__)
// binary search for CDF (cdf length = n+1). returns index in [0..n-1]
inline __device__ int binarySearchCdf(float const* cdf, int n, float u)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi)
    {
        int   mid = (lo + hi) >> 1;
        float v   = cdf[mid + 1]; // upper edge for cell mid
        if (u < v)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    int idx = lo;
    if (idx < 0)
        idx = 0;
    if (idx >= n)
        idx = n - 1;
    return idx;
}
#endif
struct FilterSampleDevice
{
    dmt::float2 p;
    float       weight;
};

// sample piecewise-constant 2D distribution on device
DMT_GPU FilterSampleDevice sampleFilter2D(DeviceFilter const& F, float u0, float u1)
{
    FilterSampleDevice out;
    int                Ny = F.Ny, Nx = F.Nx;
    float const*       marg    = F.d_marginalCdf; // Ny+1
    int                row     = binarySearchCdf(marg, Ny, u0);
    float const*       rowCdf  = F.d_condCdf + row * (Nx + 1);
    int                col     = binarySearchCdf(rowCdf, Nx, u1);
    float              condC0  = rowCdf[col];
    float              condC1  = rowCdf[col + 1];
    float              condPdf = condC1 - condC0;
    float              margPdf = marg[row + 1] - marg[row];
    float              pdf     = condPdf * margPdf;
    if (pdf <= 0.0f)
        pdf = 1.0f / (float)(Nx * Ny);
    float localU1 = condPdf > 0.0f ? (u1 - condC0) / condPdf : 0.5f;
    float localU0 = margPdf > 0.0f ? (u0 - marg[row]) / margPdf : 0.5f;
    float cellW   = (F.xMax - F.xMin) / float(Nx);
    float cellH   = (F.yMax - F.yMin) / float(Ny);
    float x       = F.xMin + (col + localU1) * cellW;
    float y       = F.yMin + (row + localU0) * cellH;
    float fval    = F.d_f[row * Nx + col];
    out.p         = dmt::make_float2(x, y);
    out.weight    = fval / pdf;
    return out;
}
#endif

namespace dmt::gpu {
    /// Mitchell filter sampler for CUDA
    struct MitchellFilterGPU
    {
        float radiusX;
        float radiusY;
        float B, C;
        
    #if defined(__CUDA_ARCH__)
        __device__ float mitchell1D(float x) const
        {
            x = fabsf(x);
            if (x <= 1.0f)
            {
                return ((12 - 9 * B - 6 * C) * x * x * x + (-18 + 12 * B + 6 * C) * x * x + (6 - 2 * B)) * (1.0f / 6.0f);
            }
            else if (x <= 2.0f)
            {
                return ((-B - 6 * C) * x * x * x + (6 * B + 30 * C) * x * x + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) *
                       (1.0f / 6.0f);
            }
            return 0.0f;
        }

        __device__ float evaluate(float2 p) const
        {
            float nx = 2.0f * p.x / radiusX;
            float ny = 2.0f * p.y / radiusY;
            return mitchell1D(nx) * mitchell1D(ny);
        }

        __device__ float2 radius() const { return make_float2(radiusX, radiusY); }
    #endif
    };

    /// Pre-tabulated distribution (flattened arrays uploaded from host)
    struct FilterDistrib2D
    {
        // Flattened arrays
        float const* conditionalCdf; // size: Ny * (Nx+1)
        float const* marginalCdf;    // size: Ny+1
        int          Nx, Ny;
        
        #if defined(__CUDA_ARCH__)
        float2       pMin, pMax;
        #else
        Point2f pMin, pMax;
        #endif
        float        integral;

        
        #if defined(__CUDA_ARCH__)
        // Sample 1D from CDF (binary search)
        __device__ int sampleCdf(float const* cdf, int size, float u, float* du) const
        {
            // binary search
            int l = 0, r = size - 1;
            while (l + 1 < r)
            {
                int m = (l + r) / 2;
                if (cdf[m] <= u)
                    l = m;
                else
                    r = m;
            }
            float c0 = cdf[l];
            float c1 = cdf[l + 1];
            float t  = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0f;
            if (du)
                *du = t;
            return l;
        }

        __device__ float2 sample(float2 u, float* pdfOut = nullptr) const
        {
            // sample marginal in v (y axis)
            float duY;
            int   iy = sampleCdf(marginalCdf, Ny, u.y, &duY);

            // sample conditional in x given y
            float const* cond = &conditionalCdf[iy * (Nx + 1)];
            float        duX;
            int          ix = sampleCdf(cond, Nx, u.x, &duX);

            // map to domain
            float  fx = ((float)ix + duX) / Nx;
            float  fy = ((float)iy + duY) / Ny;
            float2 p;
            p.x = pMin.x + fx * (pMax.x - pMin.x);
            p.y = pMin.y + fy * (pMax.y - pMin.y);

            if (pdfOut)
            {
                // pdf(x,y) ≈ f(ix,iy) / integral
                float fxy = cond[ix + 1] - cond[ix]; // unnormalized function value integrated
                float px  = fxy / (marginalCdf[Ny]); // normalize
                *pdfOut   = px;
            }

            return p;
        }
        #endif
    };

    /// Combined sampler
    /// @example extern __constant__ FilterSamplerGPU d_filter; // to be filled once by host
    struct FilterSamplerGPU
    {
        MitchellFilterGPU filter;
        FilterDistrib2D   distrib;
        
    #if defined(__CUDA_ARCH__)
        __device__ float2 sample(float2 u, float* pdfOut = nullptr) const { return distrib.sample(u, pdfOut); }

        __device__ float evaluate(float2 p) const { return filter.evaluate(p); }
    #endif
    };

} // namespace dmt::gpu