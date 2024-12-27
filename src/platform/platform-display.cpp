module;

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

module platform;

namespace dmt
{

Display::Display()
{
    if (!glfwInit())
    {
        std::cerr << "Unenable to initialize glfw" << std::endl;
        return;
    }
}

Display::~Display()
{
    if (m_window != NULL)
        glfwDestroyWindow(m_window);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

void Display::ShowWindow()
{
    //init window
    Display::InitPropertyWindow();
    //redering 
    Display::PropertyWindowRenderer();
}

void Display::PropertyWindowRenderer()
{
    //state imGui window
    bool showPropertyWindow = true;

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
        glfwGetFramebufferSize(m_window, &m_displayW, &m_displayH);
        //display propertyWindow
        Display::ShowPropertyWindow(&showPropertyWindow, m_displayW, m_displayH);
        //rendering 
        ImGui::Render();
        
        glViewport(0.2f*m_displayW, 0, 0.8*m_displayW, m_displayH);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window);
    }
}

void Display::InitPropertyWindow()
{
    glfwSetErrorCallback(GlfwErrorCallback);

    //init staff for openGL
    char const* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    //obtain information about primary monitor
    m_monitor               = glfwGetPrimaryMonitor();
    m_mode = glfwGetVideoMode(m_monitor);
    glfwWindowHint(GLFW_RED_BITS, m_mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, m_mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, m_mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, m_mode->refreshRate);

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
}

//to review 
void Display::ShowPropertyWindow(bool* pOpen, int displayW, int displayH)
{
    if (scrollbar)       windowFlags |= ImGuiWindowFlags_NoScrollbar;
    if (move)            windowFlags |= ImGuiWindowFlags_NoMove;
    if (resize)          windowFlags |= ImGuiWindowFlags_NoResize;
    if (background)      windowFlags |= ImGuiWindowFlags_NoBackground;
    if (close)           pOpen = NULL; 

    mainViewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(mainViewport->WorkPos.x, mainViewport->WorkPos.y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(displayW*0.2, displayH), ImGuiCond_FirstUseEver);


    
    //create a window 
    if (!ImGui::Begin("Property DMT", pOpen, windowFlags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }
    
    
    // Most "big" widgets share a common width settings by default. See 'Demo->Layout->Widgets Width' for details.
    ImGui::PushItemWidth(ImGui::GetFontSize() * -12); 
    //Menu bar
    ShowPropertyWindowMenuBar();
    ImGui::Text("DMT Properties");
    ImGui::Spacing();
    //Header for the tracer configuration
    if(ImGui::CollapsingHeader("Configuration"))
    {
        ImGuiIO& io = ImGui::GetIO();

        if(ImGui::TreeNode("Tracer Options"))
        {
            ImGui::SeparatorText("General");
            bool check = true;
            static char str0[128] = "Hello, world!";
            ImGui::InputText("input text", str0, IM_ARRAYSIZE(str0));
            ImGui::SameLine(); Display::HelpMarker(
                "Type text\n");
            static int i0 = 123;
            ImGui::InputInt("input int", &i0);
        }
    }

    
}

void ShowMenuFile()
{
    if (ImGui::MenuItem("Open Scene", "Ctrl+O")) {}
    if (ImGui::MenuItem("Play Rendering", "Ctrl+P")) {}
    if (ImGui::MenuItem("Save Image", "Ctrl+S")) {}
    ImGui::Separator();
    if (ImGui::MenuItem("Quit", "Alt+F4")) {}
}
} // namespace dmt