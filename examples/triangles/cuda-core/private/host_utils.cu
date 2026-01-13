#include "host_utils.cuh"

#include "common_math.cuh"
#include "morton.cuh"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// private translation unit stuff
namespace {
void writePixel(uint32_t const width, uint8_t* rowMajorImage,
                float4 const* floatBuffer, uint32_t i, uint32_t const row,
                uint32_t const col);
#if DMT_ENABLE_MSE
void pixelFromMean(uint8_t* pixel, float4 mean);
void pixelFromDelta2(uint8_t* pixel, float4 d2);
#endif
}  // namespace

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

TriangleSoup triSoupFromTriangles(const HostTriangleScene& hostScene,
                                  uint32_t const bsdfCount,
                                  size_t maxTrianglesPerChunk) {
  static int constexpr ComponentCount = 4;
  TriangleSoup soup{};
  soup.count = hostScene.triangles.size();

  const size_t n = soup.count;
  const size_t bytes = n * ComponentCount * sizeof(float);

  CUDA_CHECK(cudaMalloc(&soup.xs, bytes));
  CUDA_CHECK(cudaMalloc(&soup.ys, bytes));
  CUDA_CHECK(cudaMalloc(&soup.zs, bytes));
  CUDA_CHECK(cudaMalloc(&soup.matId, soup.count * sizeof(uint32_t)));

  // Temporary host buffers (bounded)
  std::vector<float> h_x(ComponentCount * maxTrianglesPerChunk);
  std::vector<float> h_y(ComponentCount * maxTrianglesPerChunk);
  std::vector<float> h_z(ComponentCount * maxTrianglesPerChunk);
  std::vector<uint32_t> h_matId(maxTrianglesPerChunk);

  uint32_t matIdx = 0;
  uint32_t meshIdx = 0;
  uint32_t nextIncrement = hostScene.nextMeshIndices[0];

  for (size_t base = 0; base < n; base += maxTrianglesPerChunk) {
    size_t const count = std::min(maxTrianglesPerChunk, n - base);

    for (size_t i = 0; i < count; ++i) {
      if (meshIdx + 1 < hostScene.nextMeshIndices.size() &&
          base + i >= nextIncrement) {
        ++meshIdx;
        nextIncrement = hostScene.nextMeshIndices[meshIdx];
        matIdx = hostScene.meshMatIds[meshIdx];
      }

      const Triangle& t = hostScene.triangles[base + i];

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

      h_matId[i] = matIdx;
    }

    size_t const copyBytes = count * ComponentCount * sizeof(float);

    CUDA_CHECK(cudaMemcpy(soup.xs + base * ComponentCount, h_x.data(),
                          copyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(soup.ys + base * ComponentCount, h_y.data(),
                          copyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(soup.zs + base * ComponentCount, h_z.data(),
                          copyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(soup.matId + base, h_matId.data(),
                          count * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  return soup;
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

void writeOutputBufferRowMajor(float4 const* outputBuffer, uint32_t const width,
                               uint32_t const height, char const* name) {
  const auto rowMajorImage = std::make_unique<uint8_t[]>(width * height * 3);
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      uint32_t const i = row * width + col;
      writePixel(width, rowMajorImage.get(), outputBuffer, i, row, col);
    }
  }
  std::string const theOutPath = (getExecutableDirectory() / name).string();
  stbi_write_png(theOutPath.c_str(), width, height, 3, rowMajorImage.get(),
                 3 * width);
}
#if DMT_ENABLE_MSE
__host__ void writeMeanAndMSERowMajor(float4 const* mean,
                                      float4 const* deltaSqr, uint32_t width,
                                      uint32_t height, std::string baseName) {
  const auto rowMajorImage = std::make_unique<uint8_t[]>(width * height * 3);
  const auto rowMajorMSE = std::make_unique<uint8_t[]>(width * height * 3);
  std::string const meanName =
      (getExecutableDirectory() / (baseName + ".png")).string();
  std::string const MSEName =
      (getExecutableDirectory() / (baseName + "_sqrt_mse.png")).string();
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      uint32_t const i = row * width + col;
      uint8_t* meanPx = rowMajorImage.get() + i * 3;
      uint8_t* MSEPx = rowMajorMSE.get() + i * 3;

      // TODO remove
      std::cout << "L[" << row << ", " << col << "]: " << mean[i].x << " "
                << mean[i].y << " " << mean[i].z << std::endl;
      std::cout << "V[" << row << ", " << col << "]: " << deltaSqr[i].x << " "
                << deltaSqr[i].y << " " << deltaSqr[i].z << std::endl;
      std::cout << "N: " << deltaSqr[i].w << std::endl;

      pixelFromMean(meanPx, mean[i]);
      pixelFromDelta2(MSEPx, deltaSqr[i]);
    }
  }
  stbi_write_png(meanName.c_str(), width, height, 3, rowMajorImage.get(),
                 3 * width);
  stbi_write_png(MSEName.c_str(), width, height, 3, rowMajorMSE.get(),
                 3 * width);
}

static void sumAndVariance(float4& sum, float4& sum2, uint32_t N) {
  if (N == 0) {
    sum = make_float4(0, 0, 0, 0);
    sum2 = make_float4(0, 0, 0, 1);
    return;
  }

  float3 S(sum.x, sum.y, sum.z);
  float3 Q(sum2.x, sum.y, sum.y);

  float3 m = S / float(N);
  sum = make_float4(m.x, m.y, m.z, 0);

  // Biased variance (recommended)
  float3 v = Q / float(N) - m * m;

  // Clamp for numerical noise
  sum2.x = fmaxf(v.x, 0.0f);
  sum2.y = fmaxf(v.y, 0.0f);
  sum2.z = fmaxf(v.z, 0.0f);
  sum2.w = float(N);
}

__host__ void writeMeanAndMSERowMajorCompHost(float4* mean, float4* deltaSqr,
                                              uint32_t width, uint32_t height,
                                              std::string baseName,
                                              uint32_t num) {
  const auto rowMajorImage = std::make_unique<uint8_t[]>(width * height * 3);
  const auto rowMajorMSE = std::make_unique<uint8_t[]>(width * height * 3);
  std::string const meanName =
      (getExecutableDirectory() / (baseName + ".png")).string();
  std::string const MSEName =
      (getExecutableDirectory() / (baseName + "_sqrt_mse.png")).string();
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      uint32_t const i = row * width + col;
      uint8_t* meanPx = rowMajorImage.get() + i * 3;
      uint8_t* MSEPx = rowMajorMSE.get() + i * 3;
      sumAndVariance(mean[i], deltaSqr[i], num);

      // TODO remove
      std::cout << "L[" << row << ", " << col << "]: " << mean[i].x << " "
                << mean[i].y << " " << mean[i].z << std::endl;
      std::cout << "V[" << row << ", " << col << "]: " << deltaSqr[i].x << " "
                << deltaSqr[i].y << " " << deltaSqr[i].z << std::endl;
      std::cout << "N: " << deltaSqr[i].w << std::endl;

      pixelFromMean(meanPx, mean[i]);
      pixelFromDelta2(MSEPx, deltaSqr[i]);
    }
  }
  stbi_write_png(meanName.c_str(), width, height, 3, rowMajorImage.get(),
                 3 * width);
  stbi_write_png(MSEName.c_str(), width, height, 3, rowMajorMSE.get(),
                 3 * width);
}
#endif

void writeOutputBuffer(float4 const* d_outputBuffer, uint32_t const width,
                       uint32_t const height, char const* name, bool isHost) {
  MortonLayout2D const layout = mortonLayout(height, width);
  auto const rowMajorImage = std::make_unique<uint8_t[]>(width * height * 3);
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
        if (row < height && col < width) {
          writePixel(width, rowMajorImage.get(), mortonHostBuffer.get(), i, row,
                     col);
        }
      }
    } else {
      for (uint32_t i = 0; i < layout.mortonCount; ++i) {
        uint32_t row, col;
        decodeMorton2D(i, &col, &row);
        if (row < height && col < width) {
          writePixel(width, rowMajorImage.get(), d_outputBuffer, i, row, col);
        }
      }
    }
  }

  std::string const theOutPath = (getExecutableDirectory() / name).string();
  stbi_write_png(theOutPath.c_str(), width, height, 3, rowMajorImage.get(),
                 3 * width);
}

BSDF* deviceBSDF(std::vector<BSDF> const& h_bsdfs) {
  BSDF* d_bsdf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_bsdf, sizeof(BSDF) * h_bsdfs.size()));
  CUDA_CHECK(cudaMemcpy(d_bsdf, h_bsdfs.data(), sizeof(BSDF) * h_bsdfs.size(),
                        cudaMemcpyHostToDevice));

  return d_bsdf;
}

void deviceLights(std::vector<Light> const& h_lights,
                  std::vector<Light> const& h_infiniteLights, Light** d_lights,
                  Light** d_infiniteLights) {
  CUDA_CHECK(cudaMalloc(d_lights, sizeof(Light) * h_lights.size()));
  CUDA_CHECK(cudaMemcpy(*d_lights, h_lights.data(),
                        sizeof(Light) * h_lights.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMalloc(d_infiniteLights, sizeof(Light) * h_infiniteLights.size()));
  CUDA_CHECK(cudaMemcpy(*d_infiniteLights, h_infiniteLights.data(),
                        sizeof(Light) * h_infiniteLights.size(),
                        cudaMemcpyHostToDevice));
}

DeviceCamera* deviceCamera(DeviceCamera const& h_camera) {
  DeviceCamera* d_camera = nullptr;
  CUDA_CHECK(cudaMalloc(&d_camera, sizeof(DeviceCamera)));
  CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(DeviceCamera),
                        cudaMemcpyHostToDevice));
  return d_camera;
}

// conductor ior list:
// <https://chris.hindefjord.se/resources/rgb-ior-metals/>
// Gold:
// eta:   {.r = 0.18299f, .g = 0.42108f, .b = 1.37340f};
// kappa: {.r = 3.42420f, .g = 2.34590f, .b = 1.77040f};
void cornellBox(HostTriangleScene* h_scene, std::vector<Light>* h_lights,
                std::vector<Light>* h_infiniteLights,
                std::vector<BSDF>* h_bsdfs, DeviceCamera* h_camera) {
  float3 const white = make_float3(0.9f, 170.f / 204.f, 160.f / 204.f);

  *h_scene = {};
  *h_bsdfs = {};
  // ball left
  h_scene->addModel(generateSphereMesh(make_float3(-1.2, 2, -0.25), 0.5f, 2, 4),
                    0);
  // h_bsdfs->push_back(makeGGXConductor({0.18299f, 0.42108f, 1.37340f},
  //                                     {3.42420f, 2.34590f, 1.77040f}, 0.f,
  //                                     .9f, .9f));
  h_bsdfs->push_back(makeOrenNayar({1.f, .7f, .3f}, .7f));

  // ball right
  h_scene->addModel(
      generateSphereMesh(make_float3(1.2, 2.4, -0.25), 0.5f, 2, 4), 1);
  h_bsdfs->push_back(makeGGXDielectric({0.02f, 0.07f, 0.01f},
                                       {0.95f, 0.95f, 0.87f}, 1.f /*~26 deg*/,
                                       1.44f, .5f, .7f));

  // far plane
  h_scene->addModel(
      generatePlane(make_float3(0, 4, 0), make_float3(0, -1, 0), 4, 4), 2);
  h_bsdfs->push_back(makeOrenNayar(white, .5f));

  // floor plane
  h_scene->addModel(
      generatePlane(make_float3(0, 2, -.5f), make_float3(0, 0, 1), 4, 4), 3);
  h_bsdfs->push_back(makeOrenNayar({1.f, .7f, .3f}, .7f));

  // ceiling plane
  h_scene->addModel(
      generatePlane(make_float3(0, 2, 2), make_float3(0, 0, -1), 4, 4), 4);
  h_bsdfs->push_back(makeOrenNayar(white, .5f));

  // left plane
  h_scene->addModel(
      generatePlane(make_float3(-2, 2, 0), make_float3(1, 0, 0), 4, 4), 5);
  h_bsdfs->push_back(makeOrenNayar({1.f, 0.01f, 0.01f}, .6f));

  // right plane
  h_scene->addModel(
      generatePlane(make_float3(2, 2, 0), make_float3(-1, 0, 0), 4, 4), 6);
  h_bsdfs->push_back(makeOrenNayar({0.01f, 1.f, 0.01f}, .6f));

  *h_lights = {};
  h_lights->push_back(
      makeSpotLight(2.f * make_float3(1, 1, 1), make_float3(0, 1.8f, 1.7f),
                    make_float3(0, 0, -1), cosf(std::numbers::pi_v<float> / 6),
                    cosf(std::numbers::pi_v<float> / 3), 0.01f));
  *h_infiniteLights = {};
  h_infiniteLights->push_back(
      makeEnvironmentalLight(make_float3(0.1f, 0.1f, 0.1f)));

#if 0
    h_bsdfs->push_back(makeGGXConductor({0.18299f, 0.42108f, 1.37340f},
                                        {3.42420f, 2.34590f, 1.77040f}, 0.f,
                                        .9f, .9f));
    h_bsdfs->push_back(makeLambert());
#endif

  *h_camera = DeviceCamera();
  h_camera->width = 256;   // 2560;
  h_camera->height = 256;  // 1440;
  h_camera->spp = 4;
}

// private stuff impl
namespace {

#if DMT_ENABLE_MSE
void pixelFromMean(uint8_t* pixel, float4 mean) {
  float const fr = fminf(fmaxf(mean.x, 0.f) * 255.f, 255.f);
  float const fg = fminf(fmaxf(mean.y, 0.f) * 255.f, 255.f);
  float const fb = fminf(fmaxf(mean.z, 0.f) * 255.f, 255.f);
  uint8_t const u8r = static_cast<uint8_t>(fr);
  uint8_t const u8g = static_cast<uint8_t>(fg);
  uint8_t const u8b = static_cast<uint8_t>(fb);
  pixel[0] = u8r;
  pixel[1] = u8g;
  pixel[2] = u8b;
}
void pixelFromDelta2(uint8_t* pixel, float4 d2) {
  // Actually, take sqrt so that we visualize Standard Error, not MSE
  float const fr = fminf(fmaxf(safeSqrt(d2.x) / d2.w, 0.f) * 255.f, 255.f);
  float const fg = fminf(fmaxf(safeSqrt(d2.y) / d2.w, 0.f) * 255.f, 255.f);
  float const fb = fminf(fmaxf(safeSqrt(d2.z) / d2.w, 0.f) * 255.f, 255.f);
  uint8_t const u8r = static_cast<uint8_t>(fr);
  uint8_t const u8g = static_cast<uint8_t>(fg);
  uint8_t const u8b = static_cast<uint8_t>(fb);
  pixel[0] = u8r;
  pixel[1] = u8g;
  pixel[2] = u8b;
}
#endif

void writePixel(uint32_t const width, uint8_t* rowMajorImage,
                float4 const* floatBuffer, uint32_t i, uint32_t const row,
                uint32_t const col) {
  float const fr =
      fminf(fmaxf(floatBuffer[i].x / floatBuffer[i].w * 255.f, 0.f), 255.f);
  float const fg =
      fminf(fmaxf(floatBuffer[i].y / floatBuffer[i].w * 255.f, 0.f), 255.f);
  float const fb =
      fminf(fmaxf(floatBuffer[i].z / floatBuffer[i].w * 255.f, 0.f), 255.f);
  uint8_t const u8r = static_cast<uint8_t>(fr);
  uint8_t const u8g = static_cast<uint8_t>(fg);
  uint8_t const u8b = static_cast<uint8_t>(fb);
  rowMajorImage[3 * (col + row * width) + 0] = u8r;
  rowMajorImage[3 * (col + row * width) + 1] = u8g;
  rowMajorImage[3 * (col + row * width) + 2] = u8b;
  // printf("[%u %u]: %f %f %f\n", col, row, fr, fg, fb);
}

}  // namespace
