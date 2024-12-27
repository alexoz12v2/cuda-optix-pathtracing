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
        
        void Renderer();

    private:
        static void GlfwErrorCallback(int error, char const* description)
        {
            std::cerr << "GLFW Error " << error << description << std::endl;
        }
        
        void ShowPropertyWindow(bool* pOpen, int displayW, int displayH);

    private:
        GLFWwindow*  m_window;
        GLFWmonitor* m_monitor;
    };
}
