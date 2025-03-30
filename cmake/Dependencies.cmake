include(FetchContent)
include(FindPackageHandleStandardArgs)

macro(dmt_setup_dependencies)
  # link: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
  find_package(OpenGL REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  find_package(OptiX80 REQUIRED)

  if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.0 # or a later release
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(Catch2)
    message(STATUS "Put catch2 on directory ${catch2_SOURCE_DIR}")
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif()

  if(NOT TARGET Backward::Backward)
    FetchContent_Declare(backward
      GIT_REPOSITORY https://github.com/bombela/backward-cpp
      GIT_TAG master # or a version tag, such as v1.6
      SYSTEM # optional, the Backward include directory will be treated as system directory
    )
    FetchContent_MakeAvailable(backward)
  endif()

  if(NOT TARGET glad)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/glad)
  endif()

  if(NOT TARGET glm::glm)
    set(BUILD_STATIC_LIBS TRUE)
    FetchContent_Declare(glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e # refs/tags/1.0.1
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glm)
    add_compile_definitions(GLM_FORCE_XYZW_ONLY)
  endif()

  if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(Eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )

    # note: To disable eigen tests,
    # you should put this code in a add_subdirectory to avoid to change
    # BUILD_TESTING for your own project too since variables are directory
    # scoped
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    set(EIGEN_BUILD_CMAKE_PACKAGE OFF)
    FetchContent_MakeAvailable(Eigen)
  endif()

  if(NOT TARGET glfw)
    FetchContent_Declare(glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.3.4
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(glfw)
  endif()

  if(NOT TARGET Imath::Imath)
    FetchContent_Declare(Imath
      GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/Imath.git
      GIT_TAG v3.1.12 # or a later release
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(Imath)
  endif()

  if(NOT TARGET OpenEXR::OpenEXR)
    set(OPENEXR_INSTALL OFF)
    set(OPENEXR_INSTALL_TOOLS OFF)
    FetchContent_Declare(OpenEXR
      GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
      GIT_TAG v3.3.2 # or a later release
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(OpenEXR)
  endif()

  if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG v3.11.3
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(nlohmann_json)
  endif()

  # local external dependencies
  if(NOT TARGET imgui)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/imgui)
  endif()

  if(NOT TARGET implot)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/implot)
  endif()

  if(NOT TARGET stb)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/stb)
  endif()

  if(NOT TARGET qoi)
    add_subdirectory(${PROJECT_SOURCE_DIR}/extern/qoi)
  endif()
endmacro()
