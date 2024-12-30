#pragma once

#include "dmtmacros.h"

// Keep in sync with .cppm
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


DMT_MODULE_EXPORT dmt {

    typedef struct DMTwindowGLFW
    {
        GLFWwindow*  window;
        GLFWmonitor* monitor;
        const GLFWvidmode* mode;
        int displayW;
        int displayH;
        bool fullScreenState;
    }DMTwindowGLFW;

    typedef struct DMTwindowImGui
    {
        bool noScrollbar = false;
        bool noMove = true;        
        bool noResize = false;
        bool noBackground = false;
        bool menuBar = true;
        bool close = false;
        ImGuiWindowFlags windowFlags = 0;
        const ImGuiViewport* mainViewport;
    }DMTwindowImGui;

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

    private:
        static void HelpMarker(const char* desc)
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
            std::cout << "Resize: " << width << ", " << height << std::endl;
            glfwSetWindowSize(window, width, height);
            //SetFullScreen(false);
            m_winGLFW.fullScreenState = false;
        }

        static void KeyEscCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            {
                m_winGLFW.fullScreenState = false;
                //SetFullScreen(false);
                glfwSetWindowMonitor(window, NULL, 10, 10, 640, 480, 60);
	
            }
        }
        
        void ShowPropertyWindow(bool* pOpen, int displayW, int displayH);
        void ShowPropertyWindowMenuBar();
        void ShowMenuFile();
        void InitPropertyWindow();
        void PropertyWindowRenderer();

        static DMTwindowGLFW m_winGLFW;
        static DMTwindowImGui m_winImGui;
    };
}