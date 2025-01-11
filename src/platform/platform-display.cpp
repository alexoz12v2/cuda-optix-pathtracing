module;

#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

#include <cudaTest.h>

module platform;

namespace dmt
{
//glfw staff
DMTwindowGLFW   Display::m_winGLFW;
DMTwindowImGui  Display::m_winImGui;
DMTwindowOpenGL Display::m_winOpenGL;

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
        Display::ShowPropertyWindow(&showPropertyWindow);
        //rendering
        ImGui::Render();

        glViewport(0.2f * m_winGLFW.displayW, 0, 0.8 * m_winGLFW.displayW, m_winGLFW.displayH);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        dmt::RegImgSurf(m_winOpenGL.tex, m_winOpenGL.vbo, 0.2f * m_winGLFW.displayW, 0.8 * m_winGLFW.displayW);
        glUseProgram(m_winOpenGL.shaderProgram);
        glBindVertexArray(m_winOpenGL.vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_winOpenGL.tex);
        //glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

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
    m_winGLFW.monitor = glfwGetPrimaryMonitor();
    m_winGLFW.mode    = glfwGetVideoMode(m_winGLFW.monitor);
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

    int32_t version = gladLoadGL(glfwGetProcAddress);
    if (version == 0)
    {
        std::cerr << "Failed to initialize OpenGL context" << std::endl;
        return;
    }

    //set callbacks functions
    glfwSetWindowSizeCallback(m_winGLFW.window, WindowSizeCallback);
    glfwSetKeyCallback(m_winGLFW.window, KeyEscCallback);
    glfwSetWindowMaximizeCallback(m_winGLFW.window, WindowMaximizeCallback);
    //set limits minimum size window
    glfwSetWindowSizeLimits(m_winGLFW.window, 640, 480, GLFW_DONT_CARE, GLFW_DONT_CARE);

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

    Display::InitOpenGL();
}

void Display::ShowPropertyWindow(bool* pOpen)
{
    if (m_winImGui.noScrollbar)
        m_winImGui.windowFlags |= ImGuiWindowFlags_NoScrollbar;
    if (m_winImGui.noMove)
        m_winImGui.windowFlags |= ImGuiWindowFlags_NoMove;
    if (m_winImGui.noResize)
        m_winImGui.windowFlags |= ImGuiWindowFlags_NoResize;
    if (m_winImGui.noBackground)
        m_winImGui.windowFlags |= ImGuiWindowFlags_NoBackground;
    if (m_winImGui.menuBar)
        m_winImGui.windowFlags |= ImGuiWindowFlags_MenuBar;
    if (m_winImGui.alwaysAutoResize)
        m_winImGui.windowFlags |= ImGuiWindowFlags_AlwaysAutoResize;
    if (m_winImGui.noCollapse)
        m_winImGui.windowFlags |= ImGuiWindowFlags_NoCollapse;
    if (m_winImGui.close)
        pOpen = NULL;

    m_winImGui.mainViewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(m_winImGui.mainViewport->WorkPos.x, m_winImGui.mainViewport->WorkPos.y),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(m_winGLFW.displayW * 0.2, m_winGLFW.displayH), 0);


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
    if (ImGui::CollapsingHeader("Configuration"))
    {
        ImGuiIO& io = ImGui::GetIO();

        if (ImGui::TreeNode("Tracer Options"))
        {
            ImGui::SeparatorText("General");
            bool        check     = true;
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
    if (ImGui::MenuItem("Open Scene", "Ctrl+O"))
    {
    }
    if (ImGui::MenuItem("Play Rendering", "Ctrl+P"))
    {
    }
    if (ImGui::MenuItem("Save Image", "Ctrl+S"))
    {
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Quit", "Alt+F4"))
    {
        glfwSetWindowShouldClose(m_winGLFW.window, GLFW_TRUE);
    }
}

void Display::SetFullScreen(bool value)
{
    m_winGLFW.fullScreenState = value;
}

bool Display::IsFullScreen()
{
    return m_winGLFW.fullScreenState;
}

void Display::InitOpenGL()
{
    float quadVertices[16] = {
        // Positions  // TexCoords
        -1.0f,
        -1.0f,
        0.0f,
        0.0f, // Bottom-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f, // Bottom-right
        -1.0f,
        1.0f,
        0.0f,
        1.0f, // Top-left
        1.0f,
        1.0f,
        1.0f,
        1.0f // Top-right
    };

    glfwGetFramebufferSize(m_winGLFW.window, &m_winGLFW.displayW, &m_winGLFW.displayH);
    m_winOpenGL.tex = dmt::createOpenGLTexture(0.2f * m_winGLFW.displayW, 0.8f * m_winGLFW.displayH);
    if (glGetError() != GL_NO_ERROR)
    {
        std::cerr << "Could not allocate texture" << std::endl;
        return;
    }

    // Create VAO and VBO
    glGenVertexArrays(1, &(m_winOpenGL.vao));
    glGenBuffers(1, &(m_winOpenGL.vbo));
    glBindVertexArray(m_winOpenGL.vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_winOpenGL.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Vertex shader
    char const* vertexShaderSource = R"(
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

    // Fragment shader
    char const* fragmentShaderSource = R"(
        #version 460 core
        out vec4 FragColor;

        in vec2 TexCoord;
        uniform sampler2D screenTexture;

        void main()
        {
            FragColor = texture(screenTexture, TexCoord);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int  success;
    char infoLog[512];

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILES\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILES\n" << infoLog << std::endl;
    }

    m_winOpenGL.shaderProgram = glCreateProgram();
    glAttachShader(m_winOpenGL.shaderProgram, vertexShader);
    glAttachShader(m_winOpenGL.shaderProgram, fragmentShader);
    glLinkProgram(m_winOpenGL.shaderProgram);

    glGetProgramiv(m_winOpenGL.shaderProgram, GL_LINK_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(m_winOpenGL.shaderProgram, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::SHADERPROGRAM::LINKING_FAILES\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}
} // namespace dmt