module;

#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

module platform;

namespace dmt
{

Display::Display()
{
    if (!glfwInit())
        return;
}

Display::~Display()
{
    if (m_window != NULL)
        glfwDestroyWindow(m_window);
    glfwTerminate();
}
static void GlfwErrorCallback(int error, char const* description)
{
    std::cerr << "GLFW Error " << error << description << std::endl;
}

void Display::ShowWindow()
{
    glfwSetErrorCallback(GlfwErrorCallback);

    if (!glfwInit())
        return;

    //init staff for openGL
    char const* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    //obtain information about primary monitor
    m_monitor               = glfwGetPrimaryMonitor();
    GLFWvidmode const* mode = glfwGetVideoMode(m_monitor);
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

    // Create window with graphics context
    m_window = glfwCreateWindow(640, 480, "DMT", m_monitor, NULL);

    if (m_window == NULL)
        return;

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    //Steup ImGUI context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    //setup IO only keyboard
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    //Setup Dear ImGui style
    //ImGui::StyleColorDark();

    //Setup Platform/renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    //loop
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        if (glfwGetWindowAttrib(m_window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        //display propertyWindow
        Display::ShowPropertyWindow()
        //rendering 
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(m_window, &display_w, &display_h);
        glViewport(0.2f*display_w, 0, 0.8*display_w, display_h);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    //cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
}

//to review 
void Display::ShowPropertyWindow()
{
    bool scrollbar = false;
    bool move = false;
    bool close = false;
    bool resize = false;
    bool background = false;
    
    ImGuiWindowFLags windowFlags = 0;

    if (scrollbar)       window_flags |= ImGuiWindowFlags_NoScrollbar;
    if (move)            window_flags |= ImGuiWindowFlags_NoMove;
    if (resize)          window_flags |= ImGuiWindowFlags_NoResize;
    if (background)      window_flags |= ImGuiWindowFlags_NoBackground;
    if (close)           p_open = NULL; 

    //fix position and size
    const ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(mainViewport->WorkPos.x + 650, mainViewport->WorkPos.y + 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);

    
    //create a window 
    if (!ImGui::Begin("Property DMT", p_open, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }
    
    ImGui::("DEMO DEMO");
    
}
} // namespace dmt