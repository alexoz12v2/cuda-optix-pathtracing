#include "wave-kernels.cuh"

#include "cuda-core/host_utils.cuh"
#include "cuda-core/host_scene.cuh"

// TODO double buffering

WavefrontStreamInput::WavefrontStreamInput(
    uint32_t threads, uint32_t blocks, HostTriangleScene const& h_scene,
    std::vector<Light> const& h_lights,
    std::vector<Light> const& h_infiniteLights,
    std::vector<BSDF> const& h_bsdfs, DeviceCamera const& h_camera) {
  // output buffer
  d_outBuffer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_outBuffer,
                        h_camera.width * h_camera.height * sizeof(float4)));
  CUDA_CHECK(cudaMemset(d_outBuffer, 0,
                        h_camera.width * h_camera.height * sizeof(float4)));
  // GMEM queues
  static int constexpr QUEUE_CAP = 2048;
  initQueue(anyhitQueue, QUEUE_CAP);
  initQueue(closesthitQueue, QUEUE_CAP);
  initQueue(shadeQueue, QUEUE_CAP);
  initQueue(missQueue, QUEUE_CAP);

  // scene
  d_triSoup = triSoupFromTriangles(h_scene, h_bsdfs.size());
  d_bsdfs = deviceBSDF(h_bsdfs);
  deviceLights(h_lights, h_infiniteLights, &d_lights, &infiniteLights);
  lightCount = h_lights.size();
  infiniteLightCount = h_infiniteLights.size();
  d_cam = deviceCamera(h_camera);
  d_haltonOwen = copyHaltonOwenToDeviceAlloc(blocks, threads);
  sampleOffset = 0;

  // path states
  initDeviceArena(pathStateSlots, 2048);
}

WavefrontStreamInput::~WavefrontStreamInput() noexcept {
  // path states
  freeDeviceArena(pathStateSlots);

  // scene
  cudaFree(d_haltonOwen);
  cudaFree(d_triSoup.matId);
  cudaFree(d_triSoup.xs);
  cudaFree(d_triSoup.ys);
  cudaFree(d_triSoup.zs);
  cudaFree(d_bsdfs);
  cudaFree(d_lights);
  cudaFree(infiniteLights);
  cudaFree(d_cam);

  // GMEM queues
  freeQueue(anyhitQueue);
  freeQueue(closesthitQueue);
  freeQueue(shadeQueue);
  freeQueue(missQueue);

  // output buffer
  cudaFree(d_outBuffer);
}
