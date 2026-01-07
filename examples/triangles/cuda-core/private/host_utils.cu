#include "host_utils.cuh"

#include "common_math.cuh"
#include "morton.cuh"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

// private translation unit stuff
namespace {
void writeGrayscaleBMP(char const* fileName, const uint8_t* input,
                       uint32_t const width, uint32_t const height);

void writePixel(uint32_t const width, uint8_t* rowMajorImage,
                float4 const* floatBuffer, uint32_t i, uint32_t const row,
                uint32_t const col);
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
        matIdx = (matIdx + 1) % bsdfCount;
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
#if 0
      std::cout << "Pixel [" << row << ' ' << col << "] : " << outputBuffer[i].x
                << ' ' << outputBuffer[i].y << ' ' << outputBuffer[i].z << ' '
                << outputBuffer[i].w << std::endl;
#endif
      writePixel(width, rowMajorImage.get(), outputBuffer, i, row, col);
    }
  }
  std::string const theOutPath = (getExecutableDirectory() / name).string();
  writeGrayscaleBMP(theOutPath.c_str(), rowMajorImage.get(), width, height);
}

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
  writeGrayscaleBMP(theOutPath.c_str(), rowMajorImage.get(), width, height);
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
void cornellBox(bool megakernel, HostTriangleScene* h_scene,
                std::vector<Light>* h_lights,
                std::vector<Light>* h_infiniteLights,
                std::vector<BSDF>* h_bsdfs, DeviceCamera* h_camera) {
  *h_scene = {};
  // TODO test only with walls, as crash is there
  // TODO complete scene for both megakernel and wavefront
  h_scene->addModel(
      generateSphereMesh(make_float3(-1.2, 2, -0.25), 0.5f, 2, 4));
  // h_scene->addModel(generateCube(make_float3(0, 2, 0), make_float3(1, 1,
  // 1)));
  if (megakernel) {
    // far plane
    h_scene->addModel(
        generatePlane(make_float3(0, 4, 0), make_float3(0, -1, 0), 4, 4));
  }
  // floor plane
  h_scene->addModel(
      generatePlane(make_float3(0, 2, -.5f), make_float3(0, 0, 1), 4, 4));
  // ceiling plane
  h_scene->addModel(
      generatePlane(make_float3(0, 2, 2), make_float3(0, 0, -1), 4, 4));
  if (megakernel) {
    // left plane
    h_scene->addModel(
        generatePlane(make_float3(-2, 2, 0), make_float3(1, 0, 0), 4, 4));
  }
  // right plane
  h_scene->addModel(
      generatePlane(make_float3(2, 2, 0), make_float3(-1, 0, 0), 4, 4));

  *h_lights = {};
  // TODO spot light broken
  if (megakernel) {
#define SPOT_LIGHT 1
#if SPOT_LIGHT
    h_lights->push_back(makeSpotLight(
        2.f * make_float3(1, 1, 1), make_float3(0, 1.8f, 1.7f),
        make_float3(0, 0, -1), cosf(std::numbers::pi_v<float> / 6),
        cosf(std::numbers::pi_v<float> / 3), 0.01f));
#else
    h_lights->push_back(makePointLight(2.7 * make_float3(1, 1, 1),
                                       make_float3(0, 0.7f, 1.5f), 0.01f));
#endif
  } else {
    h_lights->push_back(makePointLight(4 * make_float3(1, 1, 1),
                                       make_float3(0, 0.7f, 1.5f), 0.01f));
  }
  *h_infiniteLights = {};
  h_infiniteLights->push_back(
      makeEnvironmentalLight(make_float3(0.1f, 0.1f, 0.1f)));

  *h_bsdfs = {};
  if (megakernel) {
#define ONLY_OREN_NAYAR 1
#define ONLY_LAMBERT 0
#define ONLY_CONDUCTOR 0
#define ONLY_DIELECTRIC 0
#define BSDF_ALL 0

#if BSDF_ALL || ONLY_LAMBERT
    h_bsdfs->push_back(makeLambert());
#endif
#if BSDF_ALL || ONLY_OREN_NAYAR
    // ball
    h_bsdfs->push_back(makeOrenNayar({1.f, .7f, .3f}, .7f));
#endif
#if ONLY_OREN_NAYAR
    float3 const white = make_float3(0.9f, 170.f / 204.f, 160.f / 204.f);
    // far
    h_bsdfs->push_back(makeOrenNayar(white, .5f));
    // floor
    h_bsdfs->push_back(makeOrenNayar(white, .5f));
    // ceiling
    h_bsdfs->push_back(makeOrenNayar(white, .5f));
    // left
    h_bsdfs->push_back(makeOrenNayar({1.f, 0.01f, 0.01f}, .6f));
    // right
    h_bsdfs->push_back(makeOrenNayar({0.01f, 1.f, 0.01f}, .6f));
#endif
#if BSDF_ALL || ONLY_DIELECTRIC
    h_bsdfs->push_back(makeGGXDielectric({0.2f, 0.7, 0.1f}, {0.2f, 0.7, 0.1f},
                                         1.f /*~26 deg*/, 1.44f, .5f, .7f));
#endif
#if BSDF_ALL || ONLY_CONDUCTOR
    h_bsdfs->push_back(makeGGXConductor({0.18299f, 0.42108f, 1.37340f},
                                        {3.42420f, 2.34590f, 1.77040f}, 0.f,
                                        .9f, .9f));
#endif
  } else {
    h_bsdfs->push_back(makeLambert());
  }

  *h_camera = DeviceCamera();
#if 0
  if (!megakernel) {
    h_camera->width = 16;
    h_camera->height = 16;
    h_camera->spp = 4;
  }
#endif
}

// private stuff impl
namespace {

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

}  // namespace
