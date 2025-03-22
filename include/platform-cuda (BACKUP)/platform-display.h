#pragma once

#include "dmtmacros.h"
#include <platform/platform-macros.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <atomic>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <chrono>
#include <imgui.h>
#include <iostream>
#include <numbers>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>

DMT_MODULE_EXPORT namespace dmt {
    struct DMTwindowGLFW
    {
        GLFWwindow*        window;
        GLFWmonitor*       monitor;
        GLFWvidmode const* mode;
        int                displayW;
        int                displayH;
        bool               fullScreenState = true;
    };

    struct DMTwindowImGui
    {
        ImGuiViewport const* mainViewport;
        ImGuiWindowFlags     windowFlags          = 0;
        uint32_t             noScrollbar      : 1 = false;
        uint32_t             noMove           : 1 = true;
        uint32_t             noResize         : 1 = true;
        uint32_t             noBackground     : 1 = false;
        uint32_t             menuBar          : 1 = true;
        uint32_t             close            : 1 = false;
        uint32_t             alwaysAutoResize : 1 = false;
        uint32_t             noCollapse       : 1 = true;
    };

    struct DMTwindowOpenGL
    {
        GLuint bufferID;
        GLuint tex;
        GLuint vao;
        GLuint vbo;
        GLuint shaderProgram;
    };

    class Display
    {
    public:
        Display();
        ~Display();

        //Display cannot be copied or assigned
        Display(Display const&)            = delete;
        Display& operator=(Display const&) = delete;

        //Display object cannot be moved and move assignment
        Display(Display&&)            = delete;
        Display& operator=(Display&&) = delete;

        void ShowWindow();
        void SetFullScreen(bool value);
        bool IsFullScreen();

        //void CreatePBO(GLuint& bufferID, size_t size );
        //void DeletePBO(GLuint& bufferID );

    private:
        static void HelpMarker(char const* desc)
        {
            ImGui::TextDisabled("(?)");
            if (ImGui::BeginItemTooltip())
            {
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextUnformatted(desc);
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        }

        static void GlfwErrorCallback(int error, char const* description)
        {
            std::cerr << "GLFW Error " << error << description << std::endl;
        }

        static void WindowSizeCallback(GLFWwindow* window, int width, int height)
        {
            //std::cout << "Resize: " << width << ", " << height << std::endl;
            glfwSetWindowSize(window, width, height);
        }

        static void WindowMaximizeCallback(GLFWwindow* window, int maximized)
        {
            if (maximized == GLFW_TRUE)
            {
                m_winGLFW.fullScreenState = true;
                glfwSetWindowMonitor(window,
                                     m_winGLFW.monitor,
                                     0,
                                     0,
                                     m_winGLFW.mode->width,
                                     m_winGLFW.mode->height,
                                     m_winGLFW.mode->refreshRate);
            }
        }

        static void KeyEscCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && m_winGLFW.fullScreenState)
            {
                m_winGLFW.fullScreenState = false;
                glfwSetWindowMonitor(window, NULL, 50, 50, 640, 480, m_winGLFW.mode->refreshRate);
            }
            else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && !m_winGLFW.fullScreenState)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        void ShowPropertyWindow(bool* pOpen);
        void ShowPropertyWindowMenuBar();
        void ShowMenuFile();
        void InitPropertyWindow();
        void PropertyWindowRenderer();
        void InitOpenGL();

        static DMTwindowGLFW   m_winGLFW;
        static DMTwindowImGui  m_winImGui;
        static DMTwindowOpenGL m_winOpenGL;
    };
}