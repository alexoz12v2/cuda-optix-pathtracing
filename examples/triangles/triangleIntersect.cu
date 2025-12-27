#include "common.cuh"

inline uint32_t constexpr TRIANGLES_PER_THREAD = 4;
inline float constexpr MOLLER_TRUMBORE_TOLERANCE = 1e-7f;

__device__ HitResult triangleIntersect(float4 x, float4 y, float4 z, Ray ray) {
  HitResult result;

  float3 const e0{x.y - x.x, y.y - y.x, z.y - z.x};
  float3 const e1{x.z - x.x, y.z - y.x, z.z - z.x};

  float3 const d_cross_e1{ray.d.y * e1.z - ray.d.z * e1.y,
                          ray.d.z * e1.x - ray.d.x * e1.z,
                          ray.d.x * e1.y - ray.d.y * e1.x};

  float const det =
      d_cross_e1.x * e0.x + d_cross_e1.y * e0.y + d_cross_e1.z * e0.z;
  if (fabsf(det) < MOLLER_TRUMBORE_TOLERANCE) return result;  // parallel

  float const invDet = 1.0f / det;
  float3 const o_to_v0{ray.o.x - x.x, ray.o.y - y.x, ray.o.z - z.x};
  float3 const t_cross_e0{o_to_v0.y * e0.z - o_to_v0.z * e0.y,
                          o_to_v0.z * e0.x - o_to_v0.x * e0.z,
                          o_to_v0.x * e0.y - o_to_v0.y * e0.x};

  float const u =
      invDet * (d_cross_e1.x * o_to_v0.x + d_cross_e1.y * o_to_v0.y +
                d_cross_e1.z * o_to_v0.z);
  float const v = invDet * (t_cross_e0.x * ray.d.x + t_cross_e0.y * ray.d.y +
                            t_cross_e0.z * ray.d.z);
  float const t = invDet * (t_cross_e0.x * e1.x + t_cross_e0.y * e1.y +
                            t_cross_e0.z * e1.z);

  bool const valid =
      (u >= -MOLLER_TRUMBORE_TOLERANCE && v >= -MOLLER_TRUMBORE_TOLERANCE &&
       (u + v) <= 1 + MOLLER_TRUMBORE_TOLERANCE) &&
      (t > 1e-4f);  // <-- reject t <= 0

  if (valid) {
    float3 const p0 = make_float3(x.x, y.x, z.x);
    float3 const p1 = make_float3(x.y, y.y, z.y);
    float3 const p2 = make_float3(x.z, y.z, z.z);

    result.hit = 1;
    result.t = t;
    result.pos = p0 + u * e0 + v * e1;
    // TODO indexed computation of normal. now gamble on counter-clockwise
    result.normal = cross(e1, e0);
    result.error = errorFromTriangleIntersection(u, v, p0, p1, p2);
  }

  return result;
}

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
      HitResult result = triangleIntersect(x, y, z, ray);
      intersected[idx + i] = result.hit ? 1 : 0;
    }
  }
}

HitResult hostIntersectMT(const float3& o, const float3& d, const float3& v0,
                          const float3& v1, const float3& v2) {
#if 0
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Ray Direction" << std::endl;
  std::cout << "    " << d.x << ", " << d.y << ", " << d.z << std::endl;
  std::cout << "Ray Origin" << std::endl;
  std::cout << "    " << o.x << ", " << o.y << ", " << o.z << std::endl;
  std::cout << "Triangle" << std::endl;
  std::cout << "    " << v0.x << ", " << v0.y << ", " << v0.z << std::endl;
  std::cout << "    " << v1.x << ", " << v1.y << ", " << v1.z << std::endl;
  std::cout << "    " << v2.x << ", " << v2.y << ", " << v2.z << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
#endif
  constexpr float EPS = 1e-7f;
  HitResult hit;

  float3 e0{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
  float3 e1{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};

  float3 p{d.y * e1.z - d.z * e1.y, d.z * e1.x - d.x * e1.z,
           d.x * e1.y - d.y * e1.x};

  float det = p.x * e0.x + p.y * e0.y + p.z * e0.z;
  if (fabs(det) < EPS) return hit;

  float invDet = 1.0f / det;

  float3 t{o.x - v0.x, o.y - v0.y, o.z - v0.z};

  float u = invDet * (p.x * t.x + p.y * t.y + p.z * t.z);
  if (u < -EPS || u > 1 + EPS) return hit;

  float3 q{t.y * e0.z - t.z * e0.y, t.z * e0.x - t.x * e0.z,
           t.x * e0.y - t.y * e0.x};

  float v = invDet * (q.x * d.x + q.y * d.y + q.z * d.z);
  if (v < -EPS || u + v > 1 + EPS) return hit;

  float tHit = invDet * (q.x * e1.x + q.y * e1.y + q.z * e1.z);
  if (tHit > EPS) {
    hit.hit = 1;
    hit.t = tHit;
    hit.pos = v0 + u * e0 + v * e1;
    // TODO indexed computation of normal. now gamble on counter-clockwise
    hit.normal = cross(e1, e0);
    hit.error = errorFromTriangleIntersection(u, v, v0, v1, v2);
  }

  return hit;
}

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