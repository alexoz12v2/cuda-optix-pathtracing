#pragma once

#include "application/application-macros.h"

#include "platform/platform-logging.h"
#include "platform/platform-context.h"
#include "platform/cuda-wrapper.h"

#include "cuda-wrappers/cuda-nvrtc.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

namespace dmt {
struct DMT_APPLICATION_API DMTwindowGLFW {
  GLFWwindow* window;
  GLFWmonitor* monitor;
  GLFWvidmode const* mode;
  int displayW;
  int displayH;
  bool fullScreenState = true;
};

struct DMT_APPLICATION_API DMTwindowImGui {
  ImGuiViewport const* mainViewport;
  ImGuiWindowFlags windowFlags = 0;
  uint32_t noScrollbar : 1 = false;
  uint32_t noMove : 1 = true;
  uint32_t noResize : 1 = true;
  uint32_t noBackground : 1 = false;
  uint32_t menuBar : 1 = true;
  uint32_t close : 1 = false;
  uint32_t alwaysAutoResize : 1 = false;
  uint32_t noCollapse : 1 = true;
};

struct DMT_APPLICATION_API DMTwindowOpenGL {
  GLuint bufferID;
  GLuint tex;
  GLuint vao;
  GLuint vbo;
  GLuint shaderProgram;
};

// TODO better
class DMT_INTERFACE DMT_APPLICATION_API Drawer {
 public:
  virtual bool draw(uint32_t tex, uint32_t buf, uint32_t width,
                    uint32_t height) = 0;
  virtual bool isValid() const = 0;
  virtual ~Drawer() noexcept = default;
};

class DMT_APPLICATION_API CUDASurfaceDrawer : public Drawer {
 public:
  explicit CUDASurfaceDrawer(
      NvcudaLibraryFunctions const* cudaApi,
      Nvrtc64_120_0LibraryFunctions const* nvrtcApi,
      std::pmr::memory_resource* resource = std::pmr::get_default_resource());
  CUDASurfaceDrawer(CUDASurfaceDrawer const&) = delete;
  CUDASurfaceDrawer(CUDASurfaceDrawer&&) noexcept;
  CUDASurfaceDrawer& operator=(CUDASurfaceDrawer const&) = delete;
  CUDASurfaceDrawer& operator=(CUDASurfaceDrawer&&) noexcept;
  ~CUDASurfaceDrawer() noexcept override;

 public:
  bool draw(uint32_t tex, uint32_t buf, uint32_t width,
            uint32_t height) override;
  bool isValid() const override;

 private:
  CUmodule m_drawModule = 0;
  CUfunction m_drawFunction = 0;
  NvcudaLibraryFunctions const* m_cudaApi = nullptr;
};

class DMT_APPLICATION_API Display {
 public:
  Display();
  ~Display();

  // Display cannot be copied or assigned
  Display(Display const&) = delete;
  Display& operator=(Display const&) = delete;

  // Display object cannot be moved and move assignment
  Display(Display&&) = delete;
  Display& operator=(Display&&) = delete;

  void ShowWindow(Drawer* drawer);
  void SetFullScreen(bool value);
  bool IsFullScreen();

  // void CreatePBO(GLuint& bufferID, size_t size );
  // void DeletePBO(GLuint& bufferID );

 private:
  static void HelpMarker(char const* desc);
  static void GlfwErrorCallback(int error, char const* description);
  static void WindowSizeCallback(GLFWwindow* window, int width, int height);
  static void WindowMaximizeCallback(GLFWwindow* window, int maximized);
  static void KeyEscCallback(GLFWwindow* window, int key, int scancode,
                             int action, int mods);

 private:
  void ShowPropertyWindow(bool* pOpen);
  void ShowPropertyWindowMenuBar();
  void ShowMenuFile();
  void InitPropertyWindow();
  void PropertyWindowRenderer(Drawer* drawer);
  void InitOpenGL();

 private:
  static DMTwindowGLFW m_winGLFW;
  static DMTwindowImGui m_winImGui;
  static DMTwindowOpenGL m_winOpenGL;
};
}  // namespace dmt