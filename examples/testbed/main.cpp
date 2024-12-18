module;

#include <GLFW/glfw3.h>

#include <atomic>
#include <chrono>
#include <imgui.h>
#include <iostream>
#include <numbers>
#include <string_view>
#include <thread>
#include <vector>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <cassert>


module platform;

namespace // all functions declared in an anonymous namespace (from the global namespace) are static by default
{


static void glfw_error_callback(int error, char const* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Main code
void WindowGUI()
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return;

    // GL 3.0 + GLSL 130
    char const* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only


    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
        return;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    // - Our Emscripten build process allows embedding fonts to be accessible at runtime from the "fonts/" folder. See Makefile.emscripten for details.
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    // Our state
    bool   show_demo_window    = true;
    bool   show_another_window = false;
    ImVec4 clear_color         = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f       = 0.0f;
            static int   counter = 0;

            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");          // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window",
                         &show_another_window); // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w,
                     clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

    void printSome(dmt::ConsoleLogger& logger)
{
    using namespace std::string_view_literals;
    logger.log("Hello World from logger");
    logger.warn("Hello Warn from logger");
    logger.error("Hello error from logger");
    logger.log("Hello World from logger");
    logger.log("Hello {} from logger", {"world"sv});
}

void testLoggingInMultithreadedEnvironment(dmt::ConsoleLogger& logger,
                                           int32_t numThreads = static_cast<int32_t>(std::thread::hardware_concurrency()))
{
    // Vector to hold all the threads
    std::vector<std::jthread> threads;

    // Atomic counter to ensure all threads are finished before printing the final result
    std::atomic<int32_t> completedThreads{0};

    // Lambda function for thread execution
    auto logTask = [&logger, &completedThreads](int32_t id)
    {
        using namespace std::string_view_literals;

        // Each thread logs different messages
        logger.log("Thread ID {}: Hello World from logger", {id});
        logger.warn("Thread ID {}: Hello Warn from logger", {id});
        logger.error("Thread ID {}: Hello error from logger", {id});
        logger.log("Thread ID {}: Hello World from logger", {id});
        logger.log("Thread ID {}: Hello {} from logger", {id, "multithreaded"sv});

        // Increment the atomic counter to signal completion
        completedThreads.fetch_add(1, std::memory_order_relaxed);
    };

    // Spawn multiple threads
    for (int32_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back(logTask, i);
    }

    // Wait for all threads to complete
    for (auto& t : threads)
    {
        t.join();
    }

    // Ensure all threads completed
    logger.log("All threads completed logging. Total threads: {}", {completedThreads.load()});
}

void TestThreadPool()
{
    // Create a thread pool with 4 threads
    dmt::ThreadPool pool(4);
    std::cout << "ThreadPool created with 4 threads." << std::endl;

    // Vector to store futures
    std::vector<std::future<int>> futures;

    // Add tasks to the thread pool
    std::cout << "Adding tasks to the ThreadPool..." << std::endl;

    for (int i = 0; i < 10; ++i)
    {
        // Add a simple task: calculate square of a number
        futures.emplace_back(pool.AddTask(
            [i]()
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
                return i * i;
            }));
    }

    // Add a task with different arguments and return types
    auto stringFuture = pool.AddTask(
        [](std::string msg, int num)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Simulate work
            return msg + " " + std::to_string(num);
        },
        "Task completed:",
        42);

    // Wait for all tasks to finish and validate results
    std::cout << "Retrieving results from tasks..." << std::endl;

    for (int i = 0; i < 10; ++i)
    {
        int result = futures[i].get();
        std::cout << "Task " << i << " result: " << result << std::endl;
        assert(result == i * i); // Verify the result is correct
    }

    std::string resultString = stringFuture.get();
    std::cout << "Special task result: " << resultString << std::endl;
    assert(resultString == "Task completed: 42");

    // Check queue size (should be 0 after all tasks are processed)
    std::cout << "Queue size: " << pool.QueueSize() << std::endl;
    assert(pool.QueueSize() == 0);

    // Shutdown the thread pool
    std::cout << "Shutting down the ThreadPool..." << std::endl;
    pool.Shutdown();
    std::cout << "ThreadPool successfully shut down." << std::endl;

    std::cout << "All tests passed!" << std::endl;
}

} // namespace

int main()
{
    dmt::CircularOStringStream oss;
    char const*                formatStr = "this is a \\{} {} string. Pi: {}, 4 pi: {}, 1000 == {}, thuthy: {}\n";
    float                      pi        = std::numbers::pi_v<float>;
    bool                       b         = true;
    int                        thou      = 1000;
    std::string_view           arg{"format"};
    oss.logInitList(formatStr, {arg, pi, dmt::StrBuf(pi, "%.5f"), thou, b});
    std::cout << oss.str() << std::endl;
    dmt::ConsoleLogger logger = dmt::ConsoleLogger::create();
    printSome(logger);
    testLoggingInMultithreadedEnvironment(logger);
    logger.trace("I shall not be seen");
    //TestThreadPool();
    WindowGUI();
}