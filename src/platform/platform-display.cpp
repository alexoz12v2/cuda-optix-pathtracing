module;

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

module platform;

namespace dmt
{
    //glfw staff
    DMTwindowGLFW Display::m_winGLFW;
    DMTwindowImGui Display::m_winImGui;

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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_winGLFW.window != NULL)
        glfwDestroyWindow(m_winGLFW.window);

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

    while (!glfwWindowShouldClose(m_winGLFW.window))
    {
        glfwPollEvents();

        if (glfwGetWindowAttrib(m_winGLFW.window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        glfwGetFramebufferSize(m_winGLFW.window, &m_winGLFW.displayW, &m_winGLFW.displayH);
        //display propertyWindow
        Display::ShowPropertyWindow(&showPropertyWindow, m_winGLFW.displayW, m_winGLFW.displayH);
        //rendering 
        ImGui::Render();
        
        glViewport(0.2f*m_winGLFW.displayW, 0, 0.8*m_winGLFW.displayW, m_winGLFW.displayH);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_winGLFW.window);
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
    m_winGLFW.monitor               = glfwGetPrimaryMonitor();
    m_winGLFW.mode = glfwGetVideoMode(m_winGLFW.monitor);
    glfwWindowHint(GLFW_RED_BITS, m_winGLFW.mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, m_winGLFW.mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, m_winGLFW.mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, m_winGLFW.mode->refreshRate);

    // Create window with graphics context
    m_winGLFW.window = glfwCreateWindow(m_winGLFW.mode->width, m_winGLFW.mode->height, "DMT", m_winGLFW.monitor, NULL);

    if (m_winGLFW.window == NULL)
        return;

    glfwMakeContextCurrent(m_winGLFW.window);
    glfwSwapInterval(1);

    glfwSetWindowSizeCallback(m_winGLFW.window, WindowSizeCallback);
    glfwSetKeyCallback(m_winGLFW.window, KeyEscCallback);
    glfwSetWindowSizeLimits(m_winGLFW.window,640, 480,GLFW_DONT_CARE, GLFW_DONT_CARE); 	
    m_winGLFW.fullScreenState = true;

    //Steup ImGUI context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    //setup IO only keyboard
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    //Setup Dear ImGui style
    ImGui::StyleColorsDark();

    //Setup Platform/renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_winGLFW.window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

//to review 
void Display::ShowPropertyWindow(bool* pOpen, int displayW, int displayH)
{
    if (m_winImGui.noScrollbar)       m_winImGui.windowFlags |= ImGuiWindowFlags_NoScrollbar;
    if (m_winImGui.noMove)            m_winImGui.windowFlags |= ImGuiWindowFlags_NoMove;
    if (m_winImGui.noResize)          m_winImGui.windowFlags |= ImGuiWindowFlags_NoResize;
    if (m_winImGui.noBackground)      m_winImGui.windowFlags |= ImGuiWindowFlags_NoBackground;
    if (m_winImGui.menuBar)           m_winImGui.windowFlags |= ImGuiWindowFlags_MenuBar;
    if (m_winImGui.close)           pOpen = NULL; 

    m_winImGui.mainViewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(m_winImGui.mainViewport->WorkPos.x, m_winImGui.mainViewport->WorkPos.y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(m_winGLFW.displayW*0.2, m_winGLFW.displayH), ImGuiCond_FirstUseEver);


    
    //create a window 
    if (!ImGui::Begin("Property DMT", pOpen, m_winImGui.windowFlags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }
    
    
    // Most "big" widgets share a common width settings by default. See 'Demo->Layout->Widgets Width' for details.
    ImGui::PushItemWidth(ImGui::GetFontSize() * -12); 
    //Menu bar
    Display::ShowPropertyWindowMenuBar();
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

            ImGui::TreePop();
            ImGui::Spacing();
        }
    } 

    ImGui::PopItemWidth();
    ImGui::End();
}

void Display::ShowPropertyWindowMenuBar()
{
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Menu"))
        {
            Display::ShowMenuFile();
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

void Display::ShowMenuFile()
{
    if (ImGui::MenuItem("Open Scene", "Ctrl+O")) {}
    if (ImGui::MenuItem("Play Rendering", "Ctrl+P")) {}
    if (ImGui::MenuItem("Save Image", "Ctrl+S")) {}
    ImGui::Separator();
    if (ImGui::MenuItem("Quit", "Alt+F4")) {}
}

void Display::SetFullScreen(bool value)
{
    m_winGLFW.fullScreenState=value;
}

bool Display::IsFullScreen()
{
    return m_winGLFW.fullScreenState;   
}
} // namespace dmt