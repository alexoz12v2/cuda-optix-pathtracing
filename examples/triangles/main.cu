#include "common.cuh"
#include "testing.h"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

#include <cassert>
#include <cstdint>
#include <memory>
#include <fstream>
#include <filesystem>
#include <string>

// - rework triangle logic such that we switch to indexed/instanced triangles
// - add lights, hence shadow ray production and casting (without lights, we
// cannot reason about bounces)
// - work on a separate executable on BSDF function and chi2 test on these (CPU
// and GPU)

// Spaces
//   Raster space
//     Pixel coordinates (x, y)
//     Origin: top-left (0,0)
//     +X → right, +Y → down
//   Camera space
//     +X → right, +Y → up, +Z → forward
//     Origin: camera pinhole
//     Units: meters
//   World/render space
//     +X → right, +Y → forward, +Z → up
//     Origin: arbitrary(preferably camera), worldFromCamera handles camera
//       orientation

// Stream Aware operations
// - for a truly asynchronous, stream-aware cudaMemcpyAsync()
//   from host to device, the source memory must be page-locked (pinned).
// - Stack memory is normally pageable, so you must register it with
//   cudaHostRegister() if you want the copy to be asynchronous.
// - You do not need to manually find page boundaries—the CUDA runtime does
//   that for you—but if you want to know how it works or align explicitly,
//   it’s OS-specific and straightforward  cudaHostRegister()
// - if possible, don't waste time pinning a whole page for a single variable.

namespace {
DeviceHaltonOwen* copyHaltonOwenToDeviceAlloc(uint32_t blocks,
                                              uint32_t threads) {
  const uint32_t warps = (blocks * threads + WARP_SIZE - 1) / WARP_SIZE;

  // Allocate host-side contiguous buffer
  std::vector<DeviceHaltonOwen> h_rng(warps);

  for (uint32_t i = 0; i < warps; ++i) {
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
      h_rng[i].dimension[lane] = (i * 33) ^ lane;
    }
  }

  // Allocate device memory
  DeviceHaltonOwen* d_rng = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rng, warps * sizeof(DeviceHaltonOwen)));

  // Single synchronous copy (fast, safe)
  CUDA_CHECK(cudaMemcpy(d_rng, h_rng.data(), warps * sizeof(DeviceHaltonOwen),
                        cudaMemcpyHostToDevice));

  return d_rng;
}

TriangleSoup triSoupFromTriangles(const std::vector<Triangle>& tris,
                                  size_t maxTrianglesPerChunk = 1'000'000) {
  static int constexpr ComponentCount = 4;
  TriangleSoup soup{};
  soup.count = tris.size();

  const size_t n = soup.count;
  const size_t bytes = n * ComponentCount * sizeof(float);

  CUDA_CHECK(cudaMalloc(&soup.xs, bytes));
  CUDA_CHECK(cudaMalloc(&soup.ys, bytes));
  CUDA_CHECK(cudaMalloc(&soup.zs, bytes));

  // Temporary host buffers (bounded)
  std::vector<float> h_x(ComponentCount * maxTrianglesPerChunk);
  std::vector<float> h_y(ComponentCount * maxTrianglesPerChunk);
  std::vector<float> h_z(ComponentCount * maxTrianglesPerChunk);

  for (size_t base = 0; base < n; base += maxTrianglesPerChunk) {
    size_t count = std::min(maxTrianglesPerChunk, n - base);

    for (size_t i = 0; i < count; ++i) {
      const Triangle& t = tris[base + i];

      h_x[i * ComponentCount + 0] = t.v0.x;
      h_x[i * ComponentCount + 1] = t.v1.x;
      h_x[i * ComponentCount + 2] = t.v2.x;
      h_x[i * ComponentCount + 3] = 0;

      h_y[i * ComponentCount + 0] = t.v0.y;
      h_y[i * ComponentCount + 1] = t.v1.y;
      h_y[i * ComponentCount + 2] = t.v2.y;
      h_y[i * ComponentCount + 3] = 0;

      h_z[i * ComponentCount + 0] = t.v0.z;
      h_z[i * ComponentCount + 1] = t.v1.z;
      h_z[i * ComponentCount + 2] = t.v2.z;
      h_z[i * ComponentCount + 3] = 0;
    }

    size_t copyBytes = count * ComponentCount * sizeof(float);

    CUDA_CHECK(cudaMemcpy(soup.xs + base * ComponentCount, h_x.data(),
                          copyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(soup.ys + base * ComponentCount, h_y.data(),
                          copyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(soup.zs + base * ComponentCount, h_z.data(),
                          copyBytes, cudaMemcpyHostToDevice));
  }

  return soup;
}

DeviceCamera* defaultDeviceCamera(uint32_t* width, uint32_t* height,
                                  DeviceCamera* outHost) {
  DeviceCamera const h_camera;
  DeviceCamera* d_camera = nullptr;
  if (outHost) *outHost = h_camera;
  CUDA_CHECK(cudaMalloc(&d_camera, sizeof(DeviceCamera)));
  CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(DeviceCamera),
                        cudaMemcpyHostToDevice));
  if (width) *width = h_camera.width;
  if (height) *height = h_camera.height;
  return d_camera;
}

float4* deviceOutputBuffer(uint32_t const width, uint32_t const height) {
  MortonLayout2D const layout = mortonLayout(height, width);
  float4* d_outputBuffer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_outputBuffer, layout.mortonCount * sizeof(float4)));
  CUDA_CHECK(
      cudaMemset(d_outputBuffer, 0, layout.mortonCount * sizeof(float4)));
  return d_outputBuffer;
}

std::filesystem::path getExecutableDirectory() {
#if defined(_WIN32)

  // Get wide path to executable
  wchar_t buffer[MAX_PATH];
  DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  if (len == 0 || len == MAX_PATH)
    throw std::runtime_error("GetModuleFileNameW failed");

  // Convert UTF-16 → UTF-8
  int utf8Len = WideCharToMultiByte(CP_UTF8, 0, buffer, len, nullptr, 0,
                                    nullptr, nullptr);

  std::string utf8Path(utf8Len, '\0');

  WideCharToMultiByte(CP_UTF8, 0, buffer, len, utf8Path.data(), utf8Len,
                      nullptr, nullptr);

  return std::filesystem::path(utf8Path).parent_path();

#elif defined(__linux__)

  char buffer[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len == -1) throw std::runtime_error("readlink failed");

  buffer[len] = '\0';
  return std::filesystem::path(buffer).parent_path();

#else
#  error Unsupported platform
#endif
}
void writeGrayscaleBMP(char const* fileName, const uint8_t* input,
                       uint32_t const width, uint32_t const height) {
  // 24-bit BMP row padding
  uint32_t const rowStride = (width * 3 + 3) & ~3u;
  uint32_t const pixelDataSize = rowStride * height;
  uint32_t const fileSize = 54 + pixelDataSize;

  uint8_t header[54] = {};
  header[0] = 'B';
  header[1] = 'M';

  memcpy(&header[2], &fileSize, 4);
  uint32_t val = 54;
  memcpy(&header[10], &val, 4);
  val = 40;
  memcpy(&header[14], &val, 4);
  memcpy(&header[18], &width, 4);
  memcpy(&header[22], &height, 4);
  uint16_t val16 = 1;
  memcpy(&header[26], &val16, 2);
  val16 = 24;
  memcpy(&header[28], &val16, 2);

  std::ofstream out(fileName, std::ios::binary);
  out.write(reinterpret_cast<char*>(header), sizeof(header));

  // Write pixels bottom-up (BMP convention)
  std::vector<uint8_t> row(rowStride, 0);

  for (uint32_t y = 0; y < height; ++y) {
    uint32_t const srcY = height - 1 - y;  // flip vertically
    const uint8_t* src = input + srcY * width * 3;
    uint8_t* dst = row.data();

    for (uint32_t x = 0; x < width; ++x) {
      dst[0] = src[3 * x + 2];  // B
      dst[1] = src[3 * x + 1];  // G
      dst[2] = src[3 * x + 0];  // R
      dst += 3;
    }

    out.write(reinterpret_cast<char*>(row.data()), rowStride);
  }
}

void writePixel(uint32_t const width, uint8_t* rowMajorImage,
                float4 const* mortonHostBuffer, uint32_t i, uint32_t const row,
                uint32_t const col) {
  float const fr = fminf(fmaxf(mortonHostBuffer[i].x * 255.f, 0.f), 255.f);
  float const fg = fminf(fmaxf(mortonHostBuffer[i].y * 255.f, 0.f), 255.f);
  float const fb = fminf(fmaxf(mortonHostBuffer[i].z * 255.f, 0.f), 255.f);
  uint8_t const u8r = static_cast<uint8_t>(fr);
  uint8_t const u8g = static_cast<uint8_t>(fg);
  uint8_t const u8b = static_cast<uint8_t>(fb);
  rowMajorImage[3 * (col + row * width) + 0] = u8r;
  rowMajorImage[3 * (col + row * width) + 1] = u8g;
  rowMajorImage[3 * (col + row * width) + 2] = u8b;
}

void writeOutputBuffer(float4 const* d_outputBuffer, uint32_t const width,
                       uint32_t const height, char const* name = "output.bmp",
                       bool isHost = false) {
  MortonLayout2D const layout = mortonLayout(height, width);
  std::unique_ptr<uint8_t[]> rowMajorImage =
      std::make_unique<uint8_t[]>(width * height * 3);
  {
    // transfer to host
    if (!isHost) {
      const auto mortonHostBuffer =
          std::make_unique<float4[]>(layout.mortonCount);
      assert(mortonHostBuffer && rowMajorImage);
      CUDA_CHECK(cudaMemcpy(mortonHostBuffer.get(), d_outputBuffer,
                            layout.mortonCount * sizeof(float4),
                            cudaMemcpyDeviceToHost));
      for (uint32_t i = 0; i < layout.mortonCount; ++i) {
        uint32_t row, col;
        decodeMorton2D(i, &col, &row);
        writePixel(width, rowMajorImage.get(), mortonHostBuffer.get(), i, row,
                   col);
      }
    } else {
      for (uint32_t i = 0; i < layout.mortonCount; ++i) {
        uint32_t row, col;
        decodeMorton2D(i, &col, &row);
        writePixel(width, rowMajorImage.get(), d_outputBuffer, i, row, col);
      }
    }
  }

  std::string const theOutPath = (getExecutableDirectory() / name).string();
  writeGrayscaleBMP(theOutPath.c_str(), rowMajorImage.get(), width, height);
}

Ray cameraToPixelCenterRay(int2 pixel, Transform const& cameraFromRaster,
                           Transform const& renderFromCamera) {
  Ray ray{};

  const auto [rasterX, rasterY] = make_float2(pixel.x + 0.5f, pixel.y + 0.5f);
  float3 const pCamera =
      cameraFromRaster.apply(make_float3(rasterX, rasterY, 0.f));

  ray.o = renderFromCamera.apply(make_float3(0.0f, 0.0f, 0.0f));
  ray.d = normalize(renderFromCamera.applyDirection(pCamera));

  return ray;
}

void hostIntersectCore(int mortonOrNegative, int2 pixel,
                       Transform const& cameraFromRaster,
                       Transform const& renderFromCamera,
                       std::vector<Triangle> const& triangles,
                       std::function<void(int2, int, float)> const& onHit) {
  Ray const ray =
      cameraToPixelCenterRay(pixel, cameraFromRaster, renderFromCamera);
  HitResult hitResult{};
  hitResult.t = std::numeric_limits<float>::infinity();
  for (Triangle const& tri : triangles) {
    HitResult const result =
        hostIntersectMT(ray.o, ray.d, tri.v0, tri.v1, tri.v2);
    if (result.hit && result.t < hitResult.t) {
      hitResult = result;
    }
  }
  if (hitResult.hit) {
    onHit(pixel, mortonOrNegative, hitResult.t);
  }
}

void hostIntersectionKernel(
    bool morton, DeviceCamera cam, std::vector<Triangle> const& triangles,
    std::function<void(int2, int, float)> const& onHit) {
  MortonLayout2D const layout = mortonLayout(cam.height, cam.width);

  Transform const cameraFromRaster = cameraFromRaster_Perspective(
      cam.focalLength, cam.sensorSize, cam.width, cam.height);
  Transform const renderFromCamera = worldFromCamera(cam.dir, cam.pos);

  if (morton) {
    for (int i = 0; i < layout.mortonCount; ++i) {
      uint2 upixel{};
      decodeMorton2D(i, &upixel.x, &upixel.y);
      if (upixel.x >= cam.width || upixel.y >= cam.height) {
        continue;
      }

      int2 const pixel = make_int2(upixel.x, upixel.y);
      hostIntersectCore(i, pixel, cameraFromRaster, renderFromCamera, triangles,
                        onHit);
    }
  } else {
    for (int row = 0; row < cam.height; ++row) {
      for (int col = 0; col < cam.width; ++col) {
        int2 const pixel = make_int2(col, row);
        // std::cout << "PIXEL " << col << " " << row << std::endl;
        hostIntersectCore(-1, pixel, cameraFromRaster, renderFromCamera,
                          triangles, onHit);
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
namespace {

void testIntersectionMegakernel() {
#if 1
  // TODO device query for optimal sizes
  uint32_t const threads = 512;
  uint32_t const blocks = 16;
#else
  uint32_t const threads = 1;
  uint32_t const blocks = 1;
#endif
  // init scene
  std::cout << "Allocating host and device resources" << std::endl;
  uint32_t width, height;
  DeviceCamera h_camera;
  DeviceCamera* d_camera = defaultDeviceCamera(&width, &height, &h_camera);
  TriangleSoup scene{};
#if 1
  std::vector<Triangle> h_mesh =
      generateSphereMesh(make_float3(0, 1.2, 0), 0.5f, 16, 32);
#else
  std::vector<Triangle> h_mesh;
  // screen half-height at distance d = d * tan(FOV / 2)
  // FOV = 2 * atan(36 / (2 * 20)) ≈ 84°
  // h = 1 * tan(84° / 2) ≈ 0.9 m
  h_mesh.push_back({
      {-0.45f, 1.0f, -0.45f},  // bottom-left
      {0.45f, 1.0f, -0.45f},   // bottom-right
      {0.0f, 1.0f, 0.45f},     // top-center
  });
#endif
  scene = triSoupFromTriangles(h_mesh);
  DeviceHaltonOwen* d_rng = copyHaltonOwenToDeviceAlloc(blocks, threads);

#if 1
  {
    std::cout << "Computing intersection to host" << std::endl;
#  if 1
    std::vector<float4> out;
    out.resize(mortonLayout(h_camera.height, h_camera.width).mortonCount);
    hostIntersectionKernel(true, h_camera, h_mesh,
                           [&](int2 pixel, int mortonIdx, float t) {
                             float const value = std::isfinite(t) ? t / 2 : 0;
                             out[mortonIdx] = make_float4(.5f, .5f, value, 1.f);
                           });
    std::cout << "Writing to host" << std::endl;
    writeOutputBuffer(out.data(), h_camera.width, h_camera.height,
                      "output-host.bmp", true);
#  else
    std::vector<uint8_t> out;
    out.resize(h_camera.height * h_camera.width);
    memset(out.data(), 0, h_camera.height * h_camera.width);
    hostIntersectionKernel(
        false, h_camera, h_mesh, [&](int2 pixel, int _unused, float t) {
#    if 0
          if (std::isfinite(t)) {
            std::cout << "{ " << pixel.y << ", " << pixel.x
                      << " }: Intersection" << std::endl;
          }
#    endif
          float const value = std::isfinite(t) ? t / 2 : 0;
          out[pixel.x + pixel.y * h_camera.width] =
              static_cast<uint8_t>(fminf(fmaxf(value * 255.f, 0), 255.f));
        });
    std::cout << "Writing to host" << std::endl;
    std::string const outPath =
        (getExecutableDirectory() / "output-host.bmp").string();
    writeGrayscaleBMP(outPath.c_str(), out.data(), h_camera.width,
                      h_camera.height);
#  endif
  }
#endif

  // init output buffer
  float4* d_outputBuffer = deviceOutputBuffer(width, height);

  basicIntersectionMegakernel<<<blocks, threads>>>(d_camera, scene, d_rng,
                                                   d_outputBuffer);
  CUDA_CHECK(cudaGetLastError());
  std::cout << "Running CUDA Kernel" << std::endl;
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Finished, writing to file" << std::endl;

  // copy to host and to file
  writeOutputBuffer(d_outputBuffer, width, height);

  // cleanup
  std::cout << "Cleanup..." << std::endl;
  cudaFree(d_outputBuffer);
  cudaFree(d_rng);
  cudaFree(scene.xs);
  cudaFree(scene.ys);
  cudaFree(scene.zs);
  cudaFree(scene.intersected);
  cudaFree(d_camera);
}

}  // namespace

// UNICODE and _UNICODE always defined
#ifdef _WIN32
int wmain() {
#else
int main() {
#endif
#ifdef DMT_OS_WINDOWS
  SetConsoleOutputCP(CP_UTF8);
#endif

#if 0
  //device camera
  DeviceCamera h_cam;
  DeviceCamera* d_cam = nullptr;
  cudaMalloc((void**)&d_cam, sizeof(DeviceCamera));
  cudaMemcpy(d_cam, &h_cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice);
  //halton howen
  DeviceHaltonOwen h_ho;
  h_ho .32px
  DeviceHaltonOwen* d_ho = nullptr;
  cudaMalloc((void**)&d_ho, sizeof(DeviceHaltonOwen));
  cudaMemcpy(d_ho, &h_ho, sizeof(DeviceHaltonOwen), cudaMemcpyHostToDevice);
  //
  dim3 grid(1, 1, 1);
  dim3 block(32, 32, 1);
  //
  //block2dim-> buffer
  //
  //launch 
  raygenKernel<<<grid, block>>>(d_cam);
#endif
#if 1
  testIntersectionMegakernel();
#endif
}  // namespace
