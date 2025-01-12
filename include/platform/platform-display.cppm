/**
 * @file platform-display.cppm
 * @brief ...
 *
 * @defgroup platform platform Module
 * @{
 */
module;

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

export module platform:display;
export import <platform-display.h>;

/** @} */