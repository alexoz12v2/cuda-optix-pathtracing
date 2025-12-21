#ifndef DUMBTRACER_CUDA_CONSTANTS_CUH
#define DUMBTRACER_CUDA_CONSTANTS_CUH

#include "core-constants.h"
#include "cudautils/cudautils-vecmath.cuh"
#include "core-render.cuh"

namespace dmt {

    struct MitchellPiecewise2DGrid
    {
        // number of rows and columns the Piecewise 2D distribution
        // - number of rows(rows) = number of conditional 1D distributions
        // - number of columns(cols) = number of entries in each buffer, where buffer is either
        //   - conditional CDF
        //   - marginal CDF
        //   - marginal absolute value of each integral of each CDF conditional buffer (non-normalized)
        //   - conditional absolute value of generating function (non-normalized)
        // number of buffers(bufs) = (rows + 1) * 2
        // number of float = bufs * cols = (rows + 1) * 2 * cols =====> Example (rows: 64, cols: 64) => 8,320 floats => 33,280 bytes
        Point2i  gridResolution;   // 8 bytes
        Bounds2f domain;           // 16 bytes
        float    marginalIntegral; // 4 bytes
        // count(marginal CDF) = count(marginal abs_f) = rows
        // count(conditional CDF) = count(conditional abs_f) = cols
        // | marginal CDF | marginal abs_f | conditional CDF 0 | conditional abs_f 0 | ... | conditional CDF N-1 | conditional abs_f N-1 |
        float data[8320]; // 33280 bytes
        //Max domain [-2, 2]^2
        float f[MaxResolutionGrid * 4]; //8192 bytes
        //TODO: sample, and pdf method

        // __device__ filtering::FilterSample sample(Point2f u) const {}

        float DMT_CORE_API pdf(Point2f pr) const;
        //compute the mitchel f
        //note that the maximum resoulution is 64x64
        __device__ float evaluateFunction(int tid)
        {
            if (tid == 0)
            {
                //Pair check about the domain values
                this->domain.pMin = domain.pMin;
                this->domain.pMax = domain.pMax;
            }

            __syncthreads();

            //pair check
            uint32_t xSize = static_cast<uint32_t>(gridResolution.x * (domain.pMax.x - domain.pMin.x));
            uint32_t ySize = static_cast<uint32_t>(gridResolution.y * (domain.pMax.y - domain.pMin.y));

            if (tid > xSize * ySize)
                return;

            int x = tid / xSize;
            int y = tid % ySize;

            Point2f p = domain.lerp({(x + 0.5f) / xSize, (y + 0.5f) / ySize});
            float   b, c;
            b = c            = 1.f / 3;
            f[x * xSize + y] = mitchell1D(2 * p.x / domain.pMax.x, b, c) * mitchell1D(2 * p.y / domain.pMin.y, b, c);
        }

        static inline __device__ float mitchell1D(float x, float b, float c)
        {
            x = std::abs(x);
            if (x <= 1)
                return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x + (6 - 2 * b)) * (1.f / 6.f);
            else if (x <= 2)
                return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) *
                       (1.f / 6.f);
            else
                return 0;
        }
    };

    //global device constant
    extern __constant__ MitchellPiecewise2DGrid gdc_MitchellPiecewise2DGrid;
    extern __constant__ float                   gdc_EwaWeightLUT[EWA_LUT_SIZE];

} // namespace dmt

#endif