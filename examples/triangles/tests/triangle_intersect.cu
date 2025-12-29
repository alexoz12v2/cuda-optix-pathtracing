#include "triangle_intersect.cuh"

#include "cuda-core/extra_math.cuh"
#include "cuda-core/shapes.cuh"

#include <iostream>

inline uint32_t constexpr TRIANGLES_PER_THREAD = 4;

namespace {

HostTriangleSoup generateTriangleSoup(size_t count) {
  HostTriangleSoup soup;
  soup.count = count;

  soup.xs.resize(4 * count);
  soup.ys.resize(4 * count);
  soup.zs.resize(4 * count);

  for (size_t i = 0; i < count; ++i) {
    int type = i % 4;
    float base = float(i);

    float3 v0{}, v1{}, v2{};

    switch (type) {
      case 0:  // XY CCW
        v0 = {base, 0.f, 0.f};
        v1 = {base + 1.f, 0.f, 0.f};
        v2 = {base, 1.f, 0.f};
        break;
      case 1:  // XY CW
        v0 = {base, 0.f, 0.f};
        v1 = {base, 1.f, 0.f};
        v2 = {base + 1.f, 0.f, 0.f};
        break;
      case 2:  // slanted
        v0 = {base, 0.f, 0.f};
        v1 = {base + 1.f, 0.f, 0.5f};
        v2 = {base, 1.f, 0.25f};
        break;
      case 3:  // degenerate
        v0 = {base, 0.f, 0.f};
        v1 = v0;
        v2 = v0;
        break;
      default:
        break;
    }

    soup.xs[4 * i + 0] = v0.x;
    soup.xs[4 * i + 1] = v1.x;
    soup.xs[4 * i + 2] = v2.x;
    soup.xs[4 * i + 3] = 0.f;

    soup.ys[4 * i + 0] = v0.y;
    soup.ys[4 * i + 1] = v1.y;
    soup.ys[4 * i + 2] = v2.y;
    soup.ys[4 * i + 3] = 0.f;

    soup.zs[4 * i + 0] = v0.z;
    soup.zs[4 * i + 1] = v1.z;
    soup.zs[4 * i + 2] = v2.z;
    soup.zs[4 * i + 3] = 0.f;
  }

  return soup;
}

std::vector<int32_t> computeExpected(const HostTriangleSoup& soup,
                                     const Ray& ray) {
  std::vector<int32_t> expected(soup.count);

  for (size_t i = 0; i < soup.count; ++i) {
    float3 v0{soup.xs[4 * i + 0], soup.ys[4 * i + 0], soup.zs[4 * i + 0]};
    float3 v1{soup.xs[4 * i + 1], soup.ys[4 * i + 1], soup.zs[4 * i + 1]};
    float3 v2{soup.xs[4 * i + 2], soup.ys[4 * i + 2], soup.zs[4 * i + 2]};

    float3 o{ray.o.x, ray.o.y, ray.o.z};
    float3 d{ray.d.x, ray.d.y, ray.d.z};

    expected[i] = hostIntersectMT(o, d, v0, v1, v2).hit ? 1 : 0;
  }

  return expected;
}

// ---------------------------------------------------------------------------
// Test Runner
// ---------------------------------------------------------------------------
void runTestForRay(const HostTriangleSoup& hostSoup,
                   const std::vector<int32_t>& expected, const Ray& ray,
                   dim3 grid, dim3 block) {
  TriangleSoup dev{};
  uint32_t* d_intersected = nullptr;
  CUDA_CHECK(cudaMalloc(&dev.xs, hostSoup.xs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.ys, hostSoup.ys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.zs, hostSoup.zs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_intersected, hostSoup.count * sizeof(int32_t)));
  dev.count = hostSoup.count;

  CUDA_CHECK(cudaMemcpy(dev.xs, hostSoup.xs.data(),
                        hostSoup.xs.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev.ys, hostSoup.ys.data(),
                        hostSoup.ys.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev.zs, hostSoup.zs.data(),
                        hostSoup.zs.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_intersected, 0, hostSoup.count * sizeof(int32_t)));

  CudaTimer timer;
  timer.begin();
  triangleIntersectKernel<<<grid, block>>>(dev, ray, d_intersected);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = timer.end();

  std::vector<int32_t> result(hostSoup.count);
  CUDA_CHECK(cudaMemcpy(result.data(), d_intersected,
                        hostSoup.count * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < hostSoup.count; ++i) {
    if (result[i] != expected[i]) {
      std::cerr << "Mismatch at triangle " << i << std::endl;
#ifdef DMT_OS_WINDOWS
      __debugbreak();
#endif
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "Ray test | grid " << grid.x << " block " << block.x << " : "
            << ms << " ms\n";

  cudaFree(dev.xs);
  cudaFree(dev.ys);
  cudaFree(dev.zs);
  cudaFree(d_intersected);
}
}  // namespace

__global__ void triangleIntersectKernel(TriangleSoup soup, Ray ray,
                                        uint32_t* intersected) {
  int32_t const gindex = blockIdx.x * blockDim.x + threadIdx.x;
  // grid-stride loop over 4 triangles (possible template param)
  for (int32_t idx = gindex; idx < (soup.count + 3) / 4;
       idx += gridDim.x * blockDim.x * TRIANGLES_PER_THREAD) {
    for (int32_t i = 0; i < TRIANGLES_PER_THREAD && (idx + i) < soup.count;
         ++i) {
      // elem has 4 components, hence casting gets rid of shifting the idx
      float4 const x = reinterpret_cast<float4 const*>(soup.xs)[idx + i];
      float4 const y = reinterpret_cast<float4 const*>(soup.ys)[idx + i];
      float4 const z = reinterpret_cast<float4 const*>(soup.zs)[idx + i];
      HitResult const result = triangleIntersect(x, y, z, ray);
      intersected[idx + i] = result.hit ? 1 : 0;
    }
  }
}

void triangleIntersectTest() {
  constexpr size_t TRI_COUNT = 1 << 16;

  HostTriangleSoup soup = generateTriangleSoup(TRI_COUNT);

  Ray rayA{{0.25f, 0.25f, 5.f}, {0.f, 0.f, -1.f}};  // mostly hits
  Ray rayB{{-5.f, 5.f, 5.f}, {0.f, 0.f, -1.f}};     // mostly misses

  auto expectedA = computeExpected(soup, rayA);
  auto expectedB = computeExpected(soup, rayB);

  std::vector<dim3> grids = {dim3(1), dim3(32), dim3(128)};
  std::vector<dim3> blocks = {dim3(64), dim3(128), dim3(256)};

  for (auto g : grids)
    for (auto b : blocks) {
      std::cout << "Testing <<<" << g.x << ", " << b.x << ">>>" << std::endl;
      runTestForRay(soup, expectedA, rayA, g, b);
      runTestForRay(soup, expectedB, rayB, g, b);
    }

  std::cout << "All tests passed ✔\n";
}