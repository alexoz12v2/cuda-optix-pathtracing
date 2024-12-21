#include "cudaTest.h"

#include <glad/gl.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#define DMT_INTERFACE_AS_HEADER
#include <platform/platform.h>


// CUDA kernel to fill the texture with gradient data
__global__ void fillTextureKernel(uchar4* devPtr, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index     = y * width + x;
        devPtr[index] = make_uchar4(x % 256, y % 256, 128, 255); // RGBA gradient
    }
}

namespace dmt
{

uint32_t createOpenGLTexture(int width, int height)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    return texture;
}

bool RegImg(uint32_t tex, uint32_t width, uint32_t height)
{
    cudaGraphicsResource_t ptrRes = nullptr;

    cudaError_t reMgs = cudaGraphicsGLRegisterImage(&ptrRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    if (reMgs != cudaSuccess)
        return false;

    reMgs = cudaGraphicsMapResources(1, &ptrRes, 0);

    if (reMgs != cudaSuccess)
        return false;

    //define texture array
    cudaArray_t texArray = nullptr;

    cudaGraphicsSubResourceGetMappedArray(&texArray, ptrRes, 0, 0);

    // Get a device pointer to the texture memory
    uchar4* devPtr;
    size_t  pitch;

    cudaMallocPitch(&devPtr, &pitch, width * sizeof(uchar4), height);

    // Launch the CUDA kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    fillTextureKernel<<<gridDim, blockDim>>>(devPtr, width, height);

    // Copy the CUDA memory to the OpenGL texture
    cudaMemcpyToArray(texArray, 0, 0, devPtr, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(devPtr);
    cudaGraphicsUnmapResources(1, &ptrRes, 0);

    return true;
}

inline constexpr uint32_t numConstants = 2;

union SaxpyScalarConstants_Type
{
    float    f;
    uint32_t n;
};

__constant__ SaxpyScalarConstants_Type saxpyConstants[numConstants];

/**
 * Sample CUDA device function which adds an element from array A and array B.
 */
__global__ void saxpyKernel(float const* A, float const* B, float* C)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < saxpyConstants[1].n)
    {
        C[tid] = saxpyConstants[0].f * A[tid] + B[tid];
    }
}

/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float const* A, float const* B, float scalar, float* C, uint32_t N)
{
    dmt::ConsoleLogger logger = dmt::ConsoleLogger::create();
    logger.log("Hello from a nvcc file!");

    // Launch CUDA kernel.
    float *                         d_A, *d_B, *d_C;
    SaxpyScalarConstants_Type const constants[numConstants]{{.f = scalar}, {.n = N}};

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpyToSymbol(saxpyConstants, constants, numConstants * sizeof(SaxpyScalarConstants_Type));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 const blockSize(512, 1, 1);
    dim3 const gridSize((N >> 9) + 1, 1, 1);

    saxpyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace dmt
