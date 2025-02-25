#define DMT_ENTRY_POINT
#include <platform/platform.h>
#include <platform/cudaTest.h>
#include "dummy.h"

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

inline constexpr uint32_t N = 10000;

namespace {
    // Single precision A X plus Y
    // saxpy: z_i = a \times x_i + y_i
    void cpu_saxpy_vect(float const* x, float const* y, float a, float* z, uint32_t n)
    {
        for (int i = 0; i < n; i++)
        {
            z[i] = a * x[i] + y[i];
        }
    }

    void display()
    {
        dmt::AppContextJanitor j;
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
            j.actx.error("Could not initialize the GLFW library");
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
            j.actx.error("Couldn't open window");
            return;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        int32_t version = gladLoadGL(glfwGetProcAddress);
        if (version == 0)
        {
            j.actx.error("Failed to initialize OpenGL context");
            return;
        }

        janitor.texture = dmt::createOpenGLTexture(800, 600);
        if (glGetError() != GL_NO_ERROR)
        {
            j.actx.error("Could not allocate texture");
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
            //dmt::RegImg(janitor.texture, 800, 600);
            dmt::RegImgSurf(janitor.texture, janitor.vbo, 800, 600);
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

int guardedMain()
{
    static constexpr uint32_t             threshold = 10e-4;
    std::random_device                    seed;
    std::unique_ptr<std::mt19937>         gen{std::make_unique<std::mt19937>(seed())};
    std::uniform_real_distribution<float> random;
    float                                 A[N];
    float                                 B[N];
    float                                 C[N], C_cpu[N];
    float const                           scalar = random(*gen);
    dmt::AppContext                       actx{512, 8192, {4096, 4096, 4096, 4096}};
    dmt::ctx::init(actx);

    for (int i = 0; i < N; i++)
    {
        A[i]     = random(*gen);
        B[i]     = random(*gen);
        C_cpu[i] = 0;
        C[i]     = 0;
    }

    actx.log("Starting saxpy computation on the CPU...");
    cpu_saxpy_vect(A, B, scalar, C_cpu, N);
    actx.log("Done!");

    actx.log("Starting saxpy computation on the GPU...");
    dmt::kernel(A, B, scalar, C, N);
    actx.log("Done! Showing first 4 elements of each result:");
    actx.log("CPU[0:3] = ");

    std::string str;
    for (uint32_t i = 0; i != 4; ++i)
    {
        str += std::to_string(C_cpu[i]) + ' ';
    }

    actx.log("  {}", {dmt::StrBuf(str)});
    actx.log("GPU[0:3] = ");
    str.clear();

    for (uint32_t i = 0; i != 4; ++i)
    {
        str += std::to_string(C[i]) + ' ';
    }
    actx.log("  {}\n", {dmt::StrBuf(str)});

    bool  error = false;
    float diff  = 0.0;
    for (int i = 0; i < N; i++)
    {
        diff = abs(C[i] - C_cpu[i]);
        if (diff > threshold)
        {
            error = true;
            actx.log("{} {} {} {}", {i, diff, C[i], C_cpu[i]});
        }
    }

    if (error)
        actx.log("The Results are Different!");
    else
        actx.log("The Results match!");

    actx.log("trying to create a window and fill a screen greenish?");

    std::unique_ptr<float[]> ptr = std::make_unique<float[]>(32);
    dmt::test::multiplyArr(ptr.get());

    display();

    actx.log("Programm Finished!");
    dmt::ctx::unregister();
    return 0;
}