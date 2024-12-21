// Include local CUDA header files.
#include <cudaTest.h>

// Include C++ header files.
// Clang format shouldn't mess with the order of OpenGL includes
// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h> // comes after glad
// clang-format on

#include <iostream>
#include <random>
#include <source_location>
#include <string>

import platform;

inline constexpr uint32_t N = 10000;

namespace
{
// Single precision A X plus Y
// saxpy: z_i = a \times x_i + y_i
void cpu_saxpy_vect(float const* x, float const* y, float a, float* z, uint32_t n)
{
    for (int i = 0; i < n; i++)
    {
        z[i] = a * x[i] + y[i];
    }
}

void display(dmt::ConsoleLogger& logger)
{
    class Janitor
    {
    public:
        Janitor()                              = default;
        Janitor(Janitor const&)                = delete;
        Janitor(Janitor&&) noexcept            = delete;
        Janitor& operator=(Janitor const&)     = delete;
        Janitor& operator=(Janitor&&) noexcept = delete;
        ~Janitor() noexcept
        {
            if (vao)
                glDeleteVertexArrays(1, &vao);
            if (vbo)
                glDeleteBuffers(1, &vbo);
            if (shaderProgram)
                glDeleteProgram(shaderProgram);
            if (texture)
                glDeleteTextures(1, &texture);
            if (initCalled)
                glfwTerminate();
        }

        GLuint vao = 0, vbo = 0, shaderProgram = 0, texture = 0;
        bool   initCalled = false;
    };
    Janitor janitor;

    if (!glfwInit())
    {
        logger.error("Could not initialize the GLFW library");
        return;
    }
    janitor.initCalled = true;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA GL Test", nullptr, nullptr);
    if (!window)
    {
        logger.error("Couldn't open window");
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    int32_t version = gladLoadGL(glfwGetProcAddress);
    if (version == 0)
    {
        logger.error("Failed to initialize OpenGL context");
        return;
    }

    janitor.texture = dmt::createOpenGLTexture(800, 600);
    if (glGetError() != GL_NO_ERROR)
    {
        logger.error("Could not allocate texture");
        return;
    }

    // Vertex and UV data
    // clang-format off
    float quadVertices[] = {
        // Positions  // TexCoords
        -1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
         1.0f, -1.0f, 1.0f, 0.0f, // Bottom-right
        -1.0f,  1.0f, 0.0f, 1.0f, // Top-left
         1.0f,  1.0f, 1.0f, 1.0f  // Top-right
    };
    // clang-format on

    // Create VAO and VBO
    glGenVertexArrays(1, &janitor.vao);
    glGenBuffers(1, &janitor.vbo);
    glBindVertexArray(janitor.vao);

    glBindBuffer(GL_ARRAY_BUFFER, janitor.vbo);
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

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    janitor.shaderProgram = glCreateProgram();
    glAttachShader(janitor.shaderProgram, vertexShader);
    glAttachShader(janitor.shaderProgram, fragmentShader);
    glLinkProgram(janitor.shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Render loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        glClearColor(0.7f, 0.5f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // CUDA!
        dmt::RegImg(janitor.texture, 800, 600);

        glUseProgram(janitor.shaderProgram);
        glBindVertexArray(janitor.vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, janitor.texture);
        glUniform1i(glGetUniformLocation(janitor.shaderProgram, "screenTexture"), 0);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
    }
}
} // namespace

int main()
{
    static constexpr uint32_t             threshold = 10e-4;
    std::random_device                    seed;
    std::mt19937                          gen{seed()};
    std::uniform_real_distribution<float> random;
    float                                 A[N];
    float                                 B[N];
    float                                 C[N], C_cpu[N];
    float const                           scalar = random(gen);
    dmt::ConsoleLogger                    logger = dmt::ConsoleLogger::create();

    for (int i = 0; i < N; i++)
    {
        A[i]     = random(gen);
        B[i]     = random(gen);
        C_cpu[i] = 0;
        C[i]     = 0;
    }

    logger.log("Starting saxpy computation on the CPU...");
    cpu_saxpy_vect(A, B, scalar, C_cpu, N);
    logger.log("Done!");

    logger.log("Starting saxpy computation on the GPU...");
    dmt::kernel(A, B, scalar, C, N);
    logger.log("Done! Showing first 4 elements of each result:");
    logger.log("CPU[0:3] = ");

    std::string str;
    for (uint32_t i = 0; i != 4; ++i)
    {
        str += std::to_string(C_cpu[i]) + ' ';
    }

    logger.log("  {}", {dmt::StrBuf(str)});
    logger.log("GPU[0:3] = ");
    str.clear();

    for (uint32_t i = 0; i != 4; ++i)
    {
        str += std::to_string(C[i]) + ' ';
    }
    logger.log("  {}\n", {dmt::StrBuf(str)});

    bool  error = false;
    float diff  = 0.0;
    for (int i = 0; i < N; i++)
    {
        diff = abs(C[i] - C_cpu[i]);
        if (diff > threshold)
        {
            error = true;
            logger.log("{} {} {} {}", {i, diff, C[i], C_cpu[i]});
        }
    }

    if (error)
        logger.log("The Results are Different!");
    else
        logger.log("The Results match!");

    logger.log("trying to create a window and fill a screen greenish?");

    display(logger);

    logger.log("Programm Finished!");
}