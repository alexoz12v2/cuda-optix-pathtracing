#pragma once

namespace dmt {
struct RenderSettings {
  // TODO
};

struct IRenderBackend {
  // scene is inglobed and converted from parsed to backend representation
  // inside the appropriate factory functions
  virtual void render(const RenderSettings&) = 0;
  virtual ~IRenderBackend() {}
};

}  // namespace dmt
