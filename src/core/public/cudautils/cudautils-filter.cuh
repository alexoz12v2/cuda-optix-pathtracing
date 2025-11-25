#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_FILTER_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_FILTER_CUH

#include "core/cudautils/cudautils-macro.cuh"
#include "core/cudautils/cudautils-vecmath.cuh"

namespace dmt::gpu {
    /// Mitchell filter sampler for CUDA
    struct MitchellFilterGPU
    {
        float radiusX;
        float radiusY;
        float B, C;

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
    };

    /// Pre-tabulated distribution (flattened arrays uploaded from host)
    struct FilterDistrib2D
    {
        // Flattened arrays
        float const* conditionalCdf; // size: Ny * (Nx+1)
        float const* marginalCdf;    // size: Ny+1
        int          Nx, Ny;

        Point2f pMin, pMax;
        float   integral;


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
    };

    /// Combined sampler
    /// @example extern __constant__ FilterSamplerGPU d_filter; // to be filled once by host
    struct FilterSamplerGPU
    {
        MitchellFilterGPU filter;
        FilterDistrib2D   distrib;

        __device__ float2 sample(float2 u, float* pdfOut = nullptr) const { return distrib.sample(u, pdfOut); }

        __device__ float evaluate(float2 p) const { return filter.evaluate(p); }
    };

} // namespace dmt::gpu
#endif // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_FILTER_CUH
