#pragma once

#include "cudautils/cudautils-macro.h"
#include "cudautils/cudautils.h"

struct DeviceFilter
{
    float* d_f;           // Ny * Nx
    float* d_condCdf;     // Ny * (Nx+1)
    float* d_marginalCdf; // Ny+1
    int    Nx, Ny;
    float  xMin, xMax, yMin, yMax;
    float  integral;
};

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