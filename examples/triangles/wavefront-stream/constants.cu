#include "wave-kernels.cuh"

__constant__ float CMEM_cameraFromRaster[32];
__constant__ float CMEM_renderFromCamera[32];
__constant__ DeviceHaltonOwenParams CMEM_haltonOwenParams;
__constant__ int2 CMEM_imageResolution;
__constant__ int2 CMEM_tileResolution;
__constant__ int CMEM_spp;

__host__ void allocateDeviceConstantMemory(DeviceCamera const& h_camera,
                                           int xTile, int yTile) {
  // constant memory
  allocateDeviceGGXEnergyPreservingTables();
  {
    Transform const src =
        cameraFromRaster_Perspective(h_camera.focalLength, h_camera.sensorSize,
                                     h_camera.width, h_camera.height);
    cudaMemcpyToSymbol(CMEM_cameraFromRaster, &src, sizeof(Transform), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    DeviceHaltonOwenParams const params =
        DeviceHaltonOwen::computeParams(h_camera.width, h_camera.height);
    cudaMemcpyToSymbol(CMEM_haltonOwenParams, &params,
                       sizeof(DeviceHaltonOwenParams), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    int2 const imageRes = make_int2(h_camera.width, h_camera.height);
    cudaMemcpyToSymbol(CMEM_imageResolution, &imageRes, sizeof(int2), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    int2 const tileRes = make_int2(xTile, yTile);
    cudaMemcpyToSymbol(CMEM_tileResolution, &tileRes, sizeof(int2), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    Transform const src = worldFromCamera(h_camera.dir, h_camera.pos);
    cudaMemcpyToSymbol(CMEM_renderFromCamera, &src, sizeof(Transform), 0,
                       cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(CMEM_spp, &h_camera.spp, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
}

__host__ void freeDeviceConstantMemory() {
  // constant memory
  freeDeviceGGXEnergyPreservingTables();
}
