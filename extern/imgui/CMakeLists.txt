add_library(imgui STATIC)
target_sources(
  imgui
  PUBLIC FILE_SET
         imgui_headers
         TYPE
         HEADERS
         FILES
         imgui.h
         imconfig.h
         imgui_internal.h
         imstb_rectpack.h
         imstb_textedit.h
         imstb_truetype.h
         backends/imgui_impl_glfw.h
         backends/imgui_impl_opengl3.h
  PRIVATE backends/imgui_impl_opengl3.cpp
          backends/imgui_impl_glfw.cpp
          imgui_demo.cpp
          imgui_draw.cpp
          imgui_tables.cpp
          imgui_widgets.cpp
          imgui.cpp)

target_include_directories(
  imgui
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/backends
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

target_compile_definitions(imgui PUBLIC "ImDrawIdx=unsigned int")
add_library(imgui::imgui ALIAS imgui)
