dmt_add_cli(dmt-test-cuda-kernel)
target_sources(dmt-test-cuda-kernel
  PRIVATE main.cpp cuda-test.cpp cuda-kernel.cu raygen.cu
)

set(cuda_test_kernel_files
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda-kernel.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda-queue.h
  ${CMAKE_CURRENT_SOURCE_DIR}/raygen.cu
)

set_source_files_properties(${cuda_test_kernel_files} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("CUDA Files" FILES ${cuda_test_kernel_files})

add_custom_target(dmt-cuda-test-kernel-copy-files ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders
  COMMAND ${CMAKE_COMMAND} -E copy
          ${cuda_test_kernel_files}
          ${CMAKE_BINARY_DIR}/bin/shaders
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/glm
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/_deps/glm-src/glm ${CMAKE_BINARY_DIR}/bin/shaders/glm
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/Eigen
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/_deps/eigen-src/Eigen ${CMAKE_BINARY_DIR}/bin/shaders/Eigen
  COMMENT "Copying CUDA kernel files to shaders/"
)

add_dependencies(dmt-test-cuda-kernel dmt-cuda-test-kernel-copy-files dmt-copy-cudautils)
target_include_directories(dmt-test-cuda-kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-test-cuda-kernel PRIVATE dmt::platform stb::stb dmt::cuda-wrappers dmt::core OpenEXR::OpenEXR)