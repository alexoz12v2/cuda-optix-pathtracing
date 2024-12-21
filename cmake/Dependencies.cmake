include(FetchContent)
include(FindPackageHandleStandardArgs)

macro(dmt_setup_dependencies)
  if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.0 # or a later release
      GIT_SHALLOW    ON
    )
    FetchContent_MakeAvailable(Catch2)
    message(STATUS "Put catch2 on directory ${catch2_SOURCE_DIR}")
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif()

  if(NOT TARGET fmt::fmt)
    message(STATUS "FMT should be imported with module support ${FMT_MODULE}")
    FetchContent_Declare(fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG        10.1.1 # random version, should be fine
    )
    FetchContent_MakeAvailable(fmt)
  endif()
  if(NOT TARGET glfw)
    find_package(OpenGL REQUIRED) 
    FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.4
    )
    FetchContent_MakeAvailable(glfw)
  endif()

  if(NOT TARGET glad)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/glad)
  endif()

  if(NOT TARGET imgui)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/imgui)
  endif()

  # link: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
  find_package(OpenGL REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  find_package(OptiX80 REQUIRED)
endmacro()
