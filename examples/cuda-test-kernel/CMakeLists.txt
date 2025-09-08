dmt_add_cli(dmt-test-cuda-kernel)
target_sources(dmt-test-cuda-kernel
  PRIVATE main.cpp cuda-test.cpp cuda-kernel.cu
)
set_source_files_properties(cuda-kernel.cu PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("CUDA Files" FILES cuda-kernel.cu)
set(cuda_kernel_out_file ${CMAKE_BINARY_DIR}/bin/cuda-kernel.cu)
add_custom_command(
  OUTPUT ${cuda_kernel_out_file}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cuda-kernel.cu ${CMAKE_BINARY_DIR}/bin
  COMMENT "copy file"
)
add_custom_target(copy_cuda_kernel_example ALL DEPENDS ${cuda_kernel_out_file})
add_dependencies(dmt-test-cuda-kernel copy_cuda_kernel_example)
target_include_directories(dmt-test-cuda-kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-test-cuda-kernel PRIVATE dmt::platform stb::stb dmt::cuda-wrappers OpenEXR::OpenEXR)