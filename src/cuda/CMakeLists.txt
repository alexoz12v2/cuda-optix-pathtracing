# reference: https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/rtigo9/CMakeLists.txt
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/cuda)
add_library(dmt-cuda)
target_sources(dmt-cuda
  PUBLIC
    FILE_SET dmt_cuda_headers TYPE HEADERS BASE_DIRS ${INCLUDE_DIR} FILES
      ${INCLUDE_DIR}/cudaTest.h
  PRIVATE
    cudaTest.cu
)

dmt_set_target_warnings(dmt-cuda)
dmt_set_target_optimization(dmt-cuda)
dmt_set_public_symbols_hidden(dmt-cuda)
dmt_add_compile_definitions(dmt-cuda)

target_compile_features(dmt-cuda PUBLIC cxx_std_20 PRIVATE cuda_std_20)

target_include_directories(dmt-cuda
  PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${INCLUDE_DIR}
  PRIVATE 
    ${PROJECT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX80_INCLUDE_DIR}
)

if(NOT DMT_CLANG_TIDY_COMMAND STREQUAL "")
  set_target_properties(${name} PROPERTIES CXX_CLANG_TIDY ${DMT_CLANG_TIDY_COMMAND})
endif()


target_link_libraries(dmt-cuda
  PUBLIC
    OpenGL::GL
    dmt::platform
    glad::glad 
    glfw
    imgui::imgui
  PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

add_library(dmt::cuda ALIAS dmt-cuda)