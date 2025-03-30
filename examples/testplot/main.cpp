#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#define DMT_ENTRY_POINT
#include <platform/platform.h>

#define DMT_CUDAUTILS_IMPL
#include "cudautils/cudautils.h"

namespace pbrt {
    inline float Blackbody(float lambda, float T)
    {
        if (T <= 0)
            return 0;
        float const c  = 299792458.f;
        float const h  = 6.62606957e-34f;
        float const kb = 1.3806488e-23f;
        // Return emitted radiance for blackbody at wavelength _lambda_
        float l  = lambda * 1e-9f;
        float Le = (2 * h * c * c) / (std::pow(l, 5) * (std::exp((h * c) / (l * kb * T)) - 1));
        return Le;
    }
} // namespace pbrt

// https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html
class imguiJanitor
{
public:
    imguiJanitor()
    {
        if (!glfwInit())
            return;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(800, 600, "TestPlot", nullptr, nullptr);
        if (!window)
            return;
        glfwMakeContextCurrent(window);
        int32_t version = gladLoadGL(glfwGetProcAddress);
        if (version == 0)
            return;

        glViewport(0, 0, 800, 600);
        glClearColor(0.2f, 0.2f, 0.2f, 1.f);
        glfwSwapInterval(1); // vsync

        IMGUI_CHECKVERSION();
        ctx = ImGui::CreateContext();
        if (!ctx)
            return;
        plotCtx = ImPlot::CreateContext();
        if (!plotCtx)
            return;
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init();
        m_valid = true;
    }
    imguiJanitor(imguiJanitor const&)                = delete;
    imguiJanitor(imguiJanitor&&) noexcept            = delete;
    imguiJanitor& operator=(imguiJanitor const&)     = delete;
    imguiJanitor& operator=(imguiJanitor&&) noexcept = delete;
    ~imguiJanitor() noexcept
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        glfwTerminate();
    }

    // to be called after poll events and clear
    void start()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    // to be called before swapping framebuffers
    void end()
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    bool isValid() const { return m_valid; }

    GLFWwindow*    window;
    ImGuiContext*  ctx;
    ImPlotContext* plotCtx;

private:
    bool m_valid = false;
};

int guardedMain()
{
    dmt::Ctx::init();
    struct Janitor
    {
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    // when dmt::Ctx::destroy runs, no active Context instances are allowed on the same thread, otherwise deadlock
    {
        dmt::Context ctx;
        if (!ctx.isValid())
            return 1;

        float black    = dmt::blackbody(827.f, 673.f);
        float expected = pbrt::Blackbody(827.f, 673.f);
        ctx.log("Blackbody radiation at 827 nm, 673 K is {} vs {} (W/(sr m^2))", std::make_tuple(black, expected));

        static constexpr int32_t numSamples = 1000;
        float                    xData[numSamples];
        float                    yData[numSamples];
        float                    lambda = dmt::lambdaMin();

        float const delta = (dmt::lambdaMax() - dmt::lambdaMin()) / numSamples;
        float const tempK = 4896.f;
        for (int32_t i = 0; i < 1000; ++i, lambda += delta)
        {
            xData[i] = lambda;
            yData[i] = dmt::blackbody(lambda, tempK);
        }

        // janitor artificial scope
        {
            imguiJanitor glJanitor;
            if (!glJanitor.isValid())
                return 2;

            while (!glfwWindowShouldClose(glJanitor.window))
            {
                glfwPollEvents(); // power hungry version to check events
                glClear(GL_COLOR_BUFFER_BIT);
                glJanitor.start();

                ImPlotStyle& style = ImPlot::GetStyle();
                if (ImPlot::BeginPlot("BlackBody Emission", {0, 0}))
                {
                    ImPlot::PlotLine("T=673.15 K", xData, yData, numSamples);
                    ImPlot::EndPlot();
                }

                glJanitor.end();
                glfwSwapBuffers(glJanitor.window);
            }
        }

        return 0;
    }
}