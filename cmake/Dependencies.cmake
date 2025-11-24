include(FetchContent)
include(FindPackageHandleStandardArgs)

# To Be called after setup CUDA
macro(dmt_setup_dependencies)
  # TODO BETTER: fix dependencies linknig on linux if you leave the CXX flags here, I think you can remove them from the
  # target specific options
  if (DMT_OS_LINUX)
    # Tell Clang to use your custom libc++
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if (DMT_OS_LINUX AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(LIBCXX_INCLUDE_DIR "/usr/lib/llvm-20/include/c++/v1")
        if (EXISTS "${LIBCXX_INCLUDE_DIR}")
          message(STATUS "Adding libc++ include: ${LIBCXX_INCLUDE_DIR}")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdlib++ -nostdinc++ -isystem ${LIBCXX_INCLUDE_DIR}")
        else ()
          message(WARNING "Custom libc++ include directory not found: ${LIBCXX_INCLUDE_DIR}")
        endif ()

        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
      endif ()

    endif ()
  endif ()

  set(IMATH_INSTALL OFF)

  # link: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
  find_package(OpenGL REQUIRED)
  find_package(OptiX80 REQUIRED)

  find_package(FBXSdk REQUIRED)

  if (NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.0 # or a later release
      GIT_SHALLOW ON)
    FetchContent_MakeAvailable(Catch2)
    message(STATUS "Put catch2 on directory ${catch2_SOURCE_DIR}")
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif ()

  if (NOT TARGET backward)
    FetchContent_Declare(
      backward
      GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
      GIT_TAG v1.6 # or a version tag, such as v1.6
      SYSTEM # optional, the Backward include directory will be treated as system directory
    )
    FetchContent_MakeAvailable(backward)
  endif ()

  if (NOT TARGET glad)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/glad)
  endif ()

  if (NOT TARGET glm::glm)
    set(BUILD_STATIC_LIBS TRUE)
    FetchContent_Declare(
      glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e # refs/tags/1.0.1
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glm)
    add_compile_definitions(GLM_FORCE_XYZW_ONLY)
  endif ()

  if (NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
      Eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE)

    # note: To disable eigen tests, you should put this code in a add_subdirectory to avoid to change BUILD_TESTING for
    # your own project too since variables are directory scoped
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    set(EIGEN_BUILD_CMAKE_PACKAGE OFF)
    FetchContent_MakeAvailable(Eigen)
  endif ()

  if (NOT TARGET glfw)
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.3.4
      GIT_SHALLOW ON)
    if (DMT_OS_LINUX)
      set(GLFW_BUILD_WAYLAND ${GLFW_BUILD_WAYLAND})
      message(STATUS "before make available: GLFW_BUILD_WAYLAND ${GLFW_BUILD_WAYLAND}")
    endif ()
    FetchContent_MakeAvailable(glfw)
  endif ()

  if (NOT TARGET Imath::Imath)
    FetchContent_Declare(
      Imath
      GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
      GIT_TAG v3.1.12 # or a later release
      GIT_SHALLOW ON)
    FetchContent_MakeAvailable(Imath)
  endif ()

  if (NOT TARGET OpenEXR::OpenEXR)
    set(OPENEXR_INSTALL OFF)
    set(OPENEXR_INSTALL_TOOLS OFF)
    FetchContent_Declare(
      OpenEXR
      GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
      GIT_TAG v3.3.2 # or a later release
      GIT_SHALLOW ON)
    FetchContent_MakeAvailable(OpenEXR)
    set_target_properties(OpenEXR PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  endif ()

  if (NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(
      nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG v3.11.3
      GIT_SHALLOW ON)
    FetchContent_MakeAvailable(nlohmann_json)
  endif ()

  # local external dependencies
  if (NOT TARGET imgui)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/imgui)
  endif ()

  if (NOT TARGET implot)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/implot)
  endif ()

  if (NOT TARGET stb)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/stb)
  endif ()

  if (NOT TARGET qoi)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/qoi)
  endif ()

  if (NOT TARGET gx)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/gx)
  endif ()
endmacro()


# Reference: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
function(dmt_setup_cuda_toolkit_11_8)
  # we want to control search completely, hence don't use stuff from the system
  unset(ENV{CUDA_PATH})
  unset(CACHE{CMAKE_CUDA_COMPILER})
  unset(CUDACXX)
  unset(CACHE{CUDAToolkit_BIN_DIR})
  unset(CACHE{CUDAToolkit_NVCC_EXECUTABLE})
  set(CUDA_VERSION 12.6)
  if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(CUDAToolkit_ROOT "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v${CUDA_VERSION}" CACHE PATH "CUDA Toolkit" FORCE)
    set(CUDAToolkit_BIN_DIR "${CUDAToolkit_ROOT}\\bin" CACHE PATH "CUDA Binary Directory" FORCE)
    set(CUDAToolkit_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}\\nvcc.exe" CACHE PATH "CUDA nvcc Compiler Driver" FORCE)
  elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(CUDAToolkit_ROOT "/usr/local/cuda-${CUDA_VERSION}" CACHE PATH "CUDA Toolkit" FORCE)
    set(CUDAToolkit_BIN_DIR "${CUDAToolkit_ROOT}/bin" CACHE PATH "CUDA Binary Directory" FORCE)
    set(CUDAToolkit_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc" CACHE PATH "CUDA nvcc Compiler Driver" FORCE)
  else ()
    message(FATAL_ERROR "Unsupported OS")
  endif ()
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA nvcc Compiler Driver" FORCE)
  unset(CACHE{CMAKE_CUDA_COMPILER})
  find_program(CMAKE_CUDA_COMPILER
    nvcc
    PATHS "${CUDAToolkit_BIN_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
  #' CUDAToolkit_ROOT)
  find_package(CUDAToolkit ${CUDA_VERSION} REQUIRED EXACT)
  if (NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "Couldn't find CUDA Runtime library target")
  endif ()
  message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  # somehow nvcc 11.8 complains about cl.exe, so skip that
  if (CMAKE_CXX_COMPILER MATCHES "cl.exe$")
    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
  endif ()
  # needed MSVC v143 - VS 2022 C++ x64/x86 build tools (v14-36-17.6)
endfunction()
