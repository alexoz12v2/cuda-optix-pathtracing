#ifndef DMT_CORE_PUBLIC_CORE_TYPES_H
#define DMT_CORE_PUBLIC_CORE_TYPES_H

#include "core-macros.h"
#include "core-light.h"
#include "core-trianglemesh.h"
#include "core-texture-cache.h"

#include "cudautils/cudautils-vecmath.cuh"

#include "platform-memory.h"
#include "platform-utils.h"

#include <concepts>

namespace dmt {
struct Parameters {
  // camera parameters
  float focalLength = 20.f;
  float sensorSize = 36.f;
  Vector3f cameraDirection = {0, 1, 0};
  Vector3f cameraPosition = {0, 0, 0};
  // film parameters
  Point2i filmResolution = {128, 128};
  float gamma = 1.f;
  os::Path imagePath = os::Path::cwd() / "dmt-render.hdr";
  // sampling parameters
  int32_t samplesPerPixel = 1;
  // path tracing parameters
  int32_t maxDepth = 5;
  UniqueRef<EnvLight> envLight = nullptr;
};
static_assert(std::movable<Parameters>);

struct ParsedObject {
  Parameters params;
  MipCacheFile texCacheFiles;
  Scene scene;
};

}  // namespace dmt
#endif  // DMT_CORE_PUBLIC_CORE_TYPES_H
