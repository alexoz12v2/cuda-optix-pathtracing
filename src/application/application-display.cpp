#include "application-display.h"

#include "platform/platform.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

namespace dmt {
// glfw staff
DMTwindowGLFW Display::m_winGLFW;
DMTwindowImGui Display::m_winImGui;
DMTwindowOpenGL Display::m_winOpenGL;

static uint32_t createOpenGLTexture(int width, int height) {
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, nullptr);

  return texture;
}

Display::Display() {
  if (!glfwInit()) {
    std::cerr << "Unenable to initialize glfw" << std::endl;
    return;
  }
}

Display::~Display() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  if (m_winGLFW.window != NULL) glfwDestroyWindow(m_winGLFW.window);

  glfwTerminate();
}

void Display::ShowWindow(Drawer* drawer) {
  // init window
  Display::InitPropertyWindow();
  // redering
  Display::PropertyWindowRenderer(drawer);
}

void Display::PropertyWindowRenderer(Drawer* drawer) {
  // state imGui window
  bool showPropertyWindow = true;

  while (!glfwWindowShouldClose(m_winGLFW.window)) {
    glfwPollEvents();

    if (glfwGetWindowAttrib(m_winGLFW.window, GLFW_ICONIFIED) != 0) {
      ImGui_ImplGlfw_Sleep(10);
      continue;
    }

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    glfwGetFramebufferSize(m_winGLFW.window, &m_winGLFW.displayW,
                           &m_winGLFW.displayH);
    // display propertyWindow
    Display::ShowPropertyWindow(&showPropertyWindow);
    // rendering
    ImGui::Render();

    glViewport(0.2f * m_winGLFW.displayW, 0, 0.8 * m_winGLFW.displayW,
               m_winGLFW.displayH);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    drawer->draw(m_winOpenGL.tex, m_winOpenGL.vbo, 0.2f * m_winGLFW.displayW,
                 0.8 * m_winGLFW.displayW);
    glUseProgram(m_winOpenGL.shaderProgram);
    glBindVertexArray(m_winOpenGL.vao);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_winOpenGL.tex);
    // glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glfwSwapBuffers(m_winGLFW.window);
  }
}

void Display::InitPropertyWindow() {
  glfwSetErrorCallback(GlfwErrorCallback);
  Context ctx;

  // init staff for openGL
  char const* glsl_version = "#version 460";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  // obtain information about primary monitor
  m_winGLFW.monitor = glfwGetPrimaryMonitor();
  m_winGLFW.mode = glfwGetVideoMode(m_winGLFW.monitor);
  glfwWindowHint(GLFW_RED_BITS, m_winGLFW.mode->redBits);
  glfwWindowHint(GLFW_GREEN_BITS, m_winGLFW.mode->greenBits);
  glfwWindowHint(GLFW_BLUE_BITS, m_winGLFW.mode->blueBits);
  glfwWindowHint(GLFW_REFRESH_RATE, m_winGLFW.mode->refreshRate);

  // Create window with graphics context
  m_winGLFW.window =
      glfwCreateWindow(m_winGLFW.mode->width, m_winGLFW.mode->height, "DMT",
                       m_winGLFW.monitor, NULL);

  if (m_winGLFW.window == NULL) return;

  glfwMakeContextCurrent(m_winGLFW.window);
  glfwSwapInterval(1);

  int32_t version = gladLoadGL(glfwGetProcAddress);
  if (version == 0) {
    if (ctx.isValid()) ctx.error("Failed to initialize OpenGL context", {});
    return;
  }

  // set callbacks functions
  glfwSetWindowSizeCallback(m_winGLFW.window, WindowSizeCallback);
  glfwSetKeyCallback(m_winGLFW.window, KeyEscCallback);
  glfwSetWindowMaximizeCallback(m_winGLFW.window, WindowMaximizeCallback);
  // set limits minimum size window
  glfwSetWindowSizeLimits(m_winGLFW.window, 640, 480, GLFW_DONT_CARE,
                          GLFW_DONT_CARE);

  m_winGLFW.fullScreenState = true;

  // Steup ImGUI context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  // setup IO only keyboard
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/renderer backends
  ImGui_ImplGlfw_InitForOpenGL(m_winGLFW.window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  Display::InitOpenGL();
}

void Display::ShowPropertyWindow(bool* pOpen) {
  if (m_winImGui.noScrollbar)
    m_winImGui.windowFlags |= ImGuiWindowFlags_NoScrollbar;
  if (m_winImGui.noMove) m_winImGui.windowFlags |= ImGuiWindowFlags_NoMove;
  if (m_winImGui.noResize) m_winImGui.windowFlags |= ImGuiWindowFlags_NoResize;
  if (m_winImGui.noBackground)
    m_winImGui.windowFlags |= ImGuiWindowFlags_NoBackground;
  if (m_winImGui.menuBar) m_winImGui.windowFlags |= ImGuiWindowFlags_MenuBar;
  if (m_winImGui.alwaysAutoResize)
    m_winImGui.windowFlags |= ImGuiWindowFlags_AlwaysAutoResize;
  if (m_winImGui.noCollapse)
    m_winImGui.windowFlags |= ImGuiWindowFlags_NoCollapse;
  if (m_winImGui.close) pOpen = NULL;

  m_winImGui.mainViewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(m_winImGui.mainViewport->WorkPos.x,
                                 m_winImGui.mainViewport->WorkPos.y),
                          ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(m_winGLFW.displayW * 0.2, m_winGLFW.displayH),
                           0);

  // create a window
  if (!ImGui::Begin("Property DMT", pOpen, m_winImGui.windowFlags)) {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  // Most "big" widgets share a common width settings by default. See
  // 'Demo->Layout->Widgets Width' for details.
  ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
  // Menu bar
  Display::ShowPropertyWindowMenuBar();
  ImGui::Text("DMT Properties");
  ImGui::Spacing();
  // Header for the tracer configuration
  if (ImGui::CollapsingHeader("Configuration")) {
    ImGuiIO& io = ImGui::GetIO();

    if (ImGui::TreeNode("Tracer Options")) {
      ImGui::SeparatorText("General");
      bool check = true;
      static char str0[128] = "Hello, world!";
      ImGui::InputText("input text", str0, IM_ARRAYSIZE(str0));
      ImGui::SameLine();
      Display::HelpMarker("Type text\n");
      static int i0 = 123;
      ImGui::InputInt("input int", &i0);

      ImGui::TreePop();
      ImGui::Spacing();
    }
  }

  ImGui::PopItemWidth();
  ImGui::End();
}

void Display::ShowPropertyWindowMenuBar() {
  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("Menu")) {
      Display::ShowMenuFile();
      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
  }
}

void Display::ShowMenuFile() {
  if (ImGui::MenuItem("Open Scene", "Ctrl+O")) {
  }
  if (ImGui::MenuItem("Play Rendering", "Ctrl+P")) {
  }
  if (ImGui::MenuItem("Save Image", "Ctrl+S")) {
  }
  ImGui::Separator();
  if (ImGui::MenuItem("Quit", "Alt+F4")) {
    glfwSetWindowShouldClose(m_winGLFW.window, GLFW_TRUE);
  }
}

void Display::SetFullScreen(bool value) { m_winGLFW.fullScreenState = value; }

bool Display::IsFullScreen() { return m_winGLFW.fullScreenState; }

static char const* s_vertexShaderSource = R"(
#version 460 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
    )";

static char const* s_fragmentShaderSource = R"(
#version 460 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D screenTexture;

void main()
{
    FragColor = texture(screenTexture, TexCoord);
}
    )";

void Display::InitOpenGL() {
  struct Janitor {
    ~Janitor() {
      if (vertexShader) glDeleteShader(vertexShader);
      if (fragmentShader) glDeleteShader(fragmentShader);
    }

    GLuint vertexShader = 0;
    GLuint fragmentShader = 0;
  } j;
  // clang-format off
        float quadVertices[16] = {
            // Positions  // TexCoords
            -1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
             1.0f, -1.0f, 1.0f, 0.0f, // Bottom-right
            -1.0f,  1.0f, 0.0f, 1.0f, // Top-left
             1.0f,  1.0f, 1.0f, 1.0f  // Top-right
        };
  // clang-format on
  Context ctx;

  glfwGetFramebufferSize(m_winGLFW.window, &m_winGLFW.displayW,
                         &m_winGLFW.displayH);
  m_winOpenGL.tex = dmt::createOpenGLTexture(0.2f * m_winGLFW.displayW,
                                             0.8f * m_winGLFW.displayH);
  if (glGetError() != GL_NO_ERROR) {
    if (ctx.isValid()) ctx.error("Could not allocate texture", {});
    return;
  }

  // Create VAO and VBO
  glGenVertexArrays(1, &(m_winOpenGL.vao));
  glGenBuffers(1, &(m_winOpenGL.vbo));
  glBindVertexArray(m_winOpenGL.vao);

  glBindBuffer(GL_ARRAY_BUFFER, m_winOpenGL.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);

  j.vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(j.vertexShader, 1, &s_vertexShaderSource, nullptr);
  glCompileShader(j.vertexShader);

  int success;
  char infoLog[512];

  glGetShaderiv(j.vertexShader, GL_COMPILE_STATUS, &success);

  if (!success) {
    glGetShaderInfoLog(j.vertexShader, 512, nullptr, infoLog);
    if (ctx.isValid())
      ctx.error("ERROR::SHADER::VERTEX::COMPILATION_FAILED:\n{}",
                std::make_tuple(infoLog));
    return;
  }

  j.fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(j.fragmentShader, 1, &s_fragmentShaderSource, nullptr);
  glCompileShader(j.fragmentShader);

  glGetShaderiv(j.fragmentShader, GL_COMPILE_STATUS, &success);

  if (!success) {
    glGetShaderInfoLog(j.fragmentShader, 512, nullptr, infoLog);
    if (ctx.isValid())
      ctx.error("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED:\n{}",
                std::make_tuple(infoLog));
    return;
  }

  m_winOpenGL.shaderProgram = glCreateProgram();
  glAttachShader(m_winOpenGL.shaderProgram, j.vertexShader);
  glAttachShader(m_winOpenGL.shaderProgram, j.fragmentShader);
  glLinkProgram(m_winOpenGL.shaderProgram);

  glGetProgramiv(m_winOpenGL.shaderProgram, GL_LINK_STATUS, &success);

  if (!success) {
    glGetShaderInfoLog(m_winOpenGL.shaderProgram, 512, nullptr, infoLog);
    if (ctx.isValid())
      ctx.error("ERROR::SHADER::SHADERPROGRAM::LINKING_FAILED:\n{}",
                std::make_tuple(infoLog));
    return;
  }
}

// Static Functions ------------------------------------
void Display::HelpMarker(char const* desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::BeginItemTooltip()) {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

void Display::GlfwErrorCallback(int error, char const* description) {
  Context ctx;
  if (ctx.isValid())
    ctx.error("GLFW Error {}: {}", std::make_tuple(error, description));
}

void Display::WindowSizeCallback(GLFWwindow* window, int width, int height) {
  // std::cout << "Resize: " << width << ", " << height << std::endl;
  glfwSetWindowSize(window, width, height);
}

void Display::WindowMaximizeCallback(GLFWwindow* window, int maximized) {
  if (maximized == GLFW_TRUE) {
    m_winGLFW.fullScreenState = true;
    glfwSetWindowMonitor(window, m_winGLFW.monitor, 0, 0, m_winGLFW.mode->width,
                         m_winGLFW.mode->height, m_winGLFW.mode->refreshRate);
  }
}

void Display::KeyEscCallback(GLFWwindow* window, int key, int scancode,
                             int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS &&
      m_winGLFW.fullScreenState) {
    m_winGLFW.fullScreenState = false;
    glfwSetWindowMonitor(window, NULL, 50, 50, 640, 480,
                         m_winGLFW.mode->refreshRate);
  } else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS &&
             !m_winGLFW.fullScreenState)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// CUDASurfaceDrawer
// -----------------------------------------------------------------------------------

static char const* const s_fillAndWriteTextureKernelName =
    "fillAndWriteTextureKernelSurfObj";
static char const* const s_fillAndWriteTextureKernelSurfObj = R"(
extern "C" __global__ void fillAndWriteTextureKernelSurfObj(cudaSurfaceObject_t surfObj, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        uchar4 value = make_uchar4(x % 256, y % 256, 128, 255); // RGBA gradient
        surf2Dwrite(value, surfObj, x * sizeof(float), y);
    }
}
)";

bool CUDASurfaceDrawer::draw(uint32_t tex, uint32_t buf, uint32_t width,
                             uint32_t height) {
  CUgraphicsResource ptrRes = nullptr;
  CUresult cuRes;

  // Register the OpenGL texture with CUDA
  cuRes = m_cudaApi->cuGraphicsGLRegisterImage(&ptrRes, tex, GL_TEXTURE_2D,
                                               CU_GRAPHICS_REGISTER_FLAGS_NONE);
  if (cuRes != CUDA_SUCCESS) return false;

  // Map the resource
  cuRes = m_cudaApi->cuGraphicsMapResources(1, &ptrRes, 0);
  if (cuRes != CUDA_SUCCESS) return false;

  // Get the mapped array
  CUarray ptrArray;
  cuRes =
      m_cudaApi->cuGraphicsSubResourceGetMappedArray(&ptrArray, ptrRes, 0, 0);
  if (cuRes != CUDA_SUCCESS) return false;

  // Specify surface descriptor
  CUDA_RESOURCE_DESC resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
  resDesc.res.array.hArray = ptrArray;

  // Create surface object
  CUsurfObject surfObj;
  cuRes = m_cudaApi->cuSurfObjectCreate(&surfObj, &resDesc);
  if (cuRes != CUDA_SUCCESS) return false;

  // Launch CUDA kernel
  uint32_t const blockDim[3](16, 16, 1);
  uint32_t const gridDim[3]((width + blockDim[0] - 1) / blockDim[0],
                            (height + blockDim[1] - 1) / blockDim[1], 1);
  void* args[] = {&surfObj, &width, &height};
  // Assume we have retrieved the CUDA function handle "kernel"
  // clang-format off
        m_cudaApi->cuLaunchKernel(
            m_drawFunction, 
            gridDim[0], gridDim[1], 1, 
            blockDim[0], blockDim[1], 1, 
            0, nullptr, 
            args, nullptr
        );
  // clang-format on

  // Cleanup
  cuSurfObjectDestroy(surfObj);
  cuGraphicsUnmapResources(1, &ptrRes, 0);
  cuGraphicsUnregisterResource(ptrRes);

  return true;
}

bool CUDASurfaceDrawer::isValid() const {
  return m_drawModule != 0 && m_cudaApi != nullptr && m_drawFunction != 0;
}

CUDASurfaceDrawer::CUDASurfaceDrawer(
    NvcudaLibraryFunctions const* cudaApi,
    Nvrtc64_120_0LibraryFunctions const* nvrtcApi,
    std::pmr::memory_resource* resource)
    : m_cudaApi(cudaApi) {
  assert(cudaApi && nvrtcApi);
  Context ctx;
  if (!ctx.isValid()) return;

  nvrtcProgram fillProgram = 0;
  nvrtcResult nvrtcRes = nvrtcApi->nvrtcCreateProgram(
      &fillProgram, s_fillAndWriteTextureKernelSurfObj, "fill.cu", 0, nullptr,
      nullptr);
  if (nvrtcRes != ::NVRTC_SUCCESS) {
    ctx.error("nvrtc Failed: {}",
              std::make_tuple(nvrtcApi->nvrtcGetErrorString(nvrtcRes)));
    return;
  }

  // TODO pass compiler options (eg "G" for device debug symbols)
  nvrtcRes = nvrtcApi->nvrtcCompileProgram(fillProgram, 0, nullptr);
  if (nvrtcRes != ::NVRTC_SUCCESS) {
    if (ctx.isErrorEnabled()) {
      size_t logSize;
      nvrtcApi->nvrtcGetProgramLogSize(fillProgram, &logSize);
      std::pmr::string log{resource};
      log.resize(logSize);  // should be beyoud SSBO size
      nvrtcApi->nvrtcGetProgramLog(fillProgram, log.data());
      ctx.error("NVRTC fill kernel Compilation failed: {}",
                std::make_tuple(log));
    }
    nvrtcApi->nvrtcDestroyProgram(&fillProgram);
    return;
  }

  size_t ptxSize = 0;
  nvrtcApi->nvrtcGetPTXSize(fillProgram, &ptxSize);
  char* ptx = reinterpret_cast<char*>(resource->allocate(ptxSize));
  if (!ptx) {
    ctx.error("Couldn't allocate memory for PTX", {});
    nvrtcApi->nvrtcDestroyProgram(&fillProgram);
    return;
  }
  nvrtcApi->nvrtcGetPTX(fillProgram, ptx);
  nvrtcApi->nvrtcDestroyProgram(&fillProgram);

  if (!cudaDriverCall(m_cudaApi,
                      m_cudaApi->cuModuleLoadData(&m_drawModule, ptx))) {
    resource->deallocate(ptx, ptxSize);
    return;
  }
  if (!cudaDriverCall(m_cudaApi, m_cudaApi->cuModuleGetFunction(
                                     &m_drawFunction, m_drawModule,
                                     s_fillAndWriteTextureKernelName))) {
    m_cudaApi = nullptr;
    resource->deallocate(ptx, ptxSize);
    return;
  }
}

CUDASurfaceDrawer::CUDASurfaceDrawer(CUDASurfaceDrawer&& _that) noexcept
    : m_drawModule(std::exchange(_that.m_drawModule,
                                 static_cast<decltype(_that.m_drawModule)>(0))),
      m_drawFunction(
          std::exchange(_that.m_drawFunction,
                        static_cast<decltype(_that.m_drawFunction)>(0))),
      m_cudaApi(std::exchange(_that.m_cudaApi,
                              static_cast<decltype(_that.m_cudaApi)>(0))) {}

CUDASurfaceDrawer& CUDASurfaceDrawer::operator=(
    CUDASurfaceDrawer&& _that) noexcept {
  if (this != &_that) {
    // Release existing resources if any.
    auto cudaApi = m_cudaApi         ? m_cudaApi
                   : _that.m_cudaApi ? _that.m_cudaApi
                                     : nullptr;
    if (m_drawModule && cudaApi) cudaApi->cuModuleUnload(m_drawModule);

    // Transfer ownership of resources from _that.
    m_drawModule = std::exchange(_that.m_drawModule,
                                 static_cast<decltype(_that.m_drawModule)>(0));
    m_drawFunction = std::exchange(
        _that.m_drawFunction, static_cast<decltype(_that.m_drawFunction)>(0));
    m_cudaApi = cudaApi;
    _that.m_cudaApi = nullptr;
  }
  return *this;
}

CUDASurfaceDrawer::~CUDASurfaceDrawer() noexcept {
  if (m_drawModule && m_cudaApi) m_cudaApi->cuModuleUnload(m_drawModule);
}
}  // namespace dmt