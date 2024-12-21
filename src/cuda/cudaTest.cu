#include "cudaTest.h"

#include <glad/gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>


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

GLuint createOpenGLTexture(int width, int height)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    return texture;
}

namespace dmt
{
bool RegImg(uint32_t tex, uint32_t buf, uint32_t width, uint32_t height)
{
    cudaGraphicsResource  res;
    cudaGraphicsResource* ptrRes = &res;

    cudaError_t reMgs = cudaGraphicsGLRegisterImage(&ptrRes, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    if (reMgs != cudaSuccess)
        return false;

    reMgs = cudaGraphicsMapResources(1, ptrRes, 0);

    if (reMgs != cudaSuccess)
        return false;

    //define texture array
    cudaArray* texArray;

    cudaGraphicsSubResourceGetMappedArray(&texArray, prtRes, 0, 0);

    // Get a device pointer to the texture memory
    uchar4* devPtr;
    size_t  pitch;

    cudaMallocPitch(&devPtr, &pitch, width * sizeof(uchar4), height);

    // Launch the CUDA kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    fillTextureKernel<<<gridDim, blockDim>>>(devPtr, width, height);

    // Copy the CUDA memory to the OpenGL texture
    cudaMemcpyToArray(textureArray, 0, 0, devPtr, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(devPtr);
    cudaGraphicsUnmapResources(1, &ptrRes, 0);


    return true;
}


} // namespace dmt
