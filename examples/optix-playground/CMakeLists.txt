dmt_add_example(dmt-optix-playground)

set(display_only_files optixSphere.cu optixSphere.h)

set_source_files_properties(${display_only_files} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("CUDA files" FILES ${display_only_files})

target_sources(dmt-optix-playground PRIVATE main.cpp ${display_only_files})
target_link_libraries(dmt-optix-playground PRIVATE dmt::platform dmtcuda-Optix8 dmt::cuda-wrappers stb::stb)

set(dmt_optix_playground_cuda_files_src ${CMAKE_CURRENT_SOURCE_DIR}/optixSphere.cu
                                        ${CMAKE_CURRENT_SOURCE_DIR}/optixSphere.h)
set(dmt_optix_playground_cuda_directory ${PROJECT_BINARY_DIR}/bin/shaders)
set(dmt_optix_playground_cuda_files_dst ${dmt_optix_playground_cuda_directory})
add_custom_command(
  OUTPUT ${dmt_optix_playground_cuda_files_dst}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${dmt_optix_playground_cuda_directory}
  COMMAND ${CMAKE_COMMAND} -E copy ${dmt_optix_playground_cuda_files_src} ${dmt_optix_playground_cuda_files_dst}
  DEPENDS ${dmt_optix_playground_cuda_files_src}
  COMMENT "Copying CUDA device code files to bin directory")
add_custom_target(dmt-optix-playground-cuda-srcs ALL DEPENDS ${dmt_optix_playground_cuda_files_dst})

add_dependencies(dmt-optix-playground dmt-copy-cuda-device-code dmt-optix-playground-cuda-srcs)
