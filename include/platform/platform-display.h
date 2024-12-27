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
    }DMTwindowGLFW;

    typedef struct DMTwindowImGui
    {
        bool scrollbar = false;
        bool move = false;
        bool close = false;
        bool resize = false;
        bool background = false;
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
        
        void ShowPropertyWindow(bool* pOpen, int displayW, int displayH);
        void ShowPropertyWindowMenuBar();
        void ShowMenuFile();
        void InitPropertyWindow();
        void PropertyWindowRenderer();

    private:
        //glfw staff
        DMTwindowGLFW m_winGLFW;
        DMTwindowImGui m_winImGui;
    };
}