# Assuming CMake and Python paths are properly set
find_package(CUDAToolkit REQUIRED)

# Run python script to get CUDA DLL paths
execute_process(
  COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/scripts/find_dll_names.py -d ${CUDAToolkit_LIBRARY_ROOT} -l cudart
  OUTPUT_VARIABLE cudart_path
  ERROR_VARIABLE py_stderr
)

if(NOT py_stderr STREQUAL "")
  message(FATAL_ERROR "Script Error: ${py_stderr}")
endif()

# Remove trailing newline or spaces from cudart_path
string(STRIP ${cudart_path} cudart_path)

execute_process(
  COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/scripts/find_dll_names.py -d ${CUDAToolkit_LIBRARY_ROOT} -l nvrtc
  OUTPUT_VARIABLE nvrtc_path
  ERROR_VARIABLE py_stderr
)

if(NOT py_stderr STREQUAL "")
  message(FATAL_ERROR "Script Error: ${py_stderr}")
endif()

# Remove trailing newline or spaces from nvrtc_path
string(STRIP ${nvrtc_path} nvrtc_path)

# Copy CUDA DLLs using custom targets
add_custom_target(copy_cuda_dlls ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${cudart_path} ${CMAKE_BINARY_DIR}/bin
  COMMENT "Copying cudart.dll to the bin directory"
)

add_custom_target(copy_nvrtc_dlls ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${nvrtc_path} ${CMAKE_BINARY_DIR}/bin
  COMMENT "Copying nvrtc.dll to the bin directory"
)

# Ensure these copy steps run after the build
add_dependencies(copy_cuda_dlls dmt-platform)
add_dependencies(copy_nvrtc_dlls dmt-platform)

# Define the paths for the generated files
set(cuda_runtime_cpp_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-runtime.cpp)
set(cuda_runtime_h_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-runtime.h)
set(cuda_nvrtc_cpp_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-nvrtc.cpp)
set(cuda_nvrtc_h_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-nvrtc.h)

# Custom commands for generating wrapper files
add_custom_command(
  OUTPUT ${cuda_runtime_cpp_file} ${cuda_runtime_h_file}
  COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/scripts/generate_dll_wrapper_file.py
  "{ \"Windows\": \"${cudart_path}\", \"Linux\": \"${cudart_path}\" }"
  -i cuda_runtime_api.h cuda_gl_interop.h
  -v "_v{n}"
  -l
  -j "\"${CMAKE_SOURCE_DIR}/scripts/dll_wrapper_type_mapper_cuda_runtime.json\""
  --cpp-file "\"${cuda_runtime_cpp_file}\""
  --header-file "\"${cuda_runtime_h_file}\""
  -up
  COMMENT "Generate CUDA Runtime API Wrappers"
  VERBATIM
)

add_custom_command(
  OUTPUT ${cuda_nvrtc_cpp_file} ${cuda_nvrtc_h_file}
  COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/scripts/generate_dll_wrapper_file.py
  "{ \"Windows\": \"${nvrtc_path}\", \"Linux\": \"${nvrtc_path}\" }"
  -i nvrtc.h
  -v "_v{n}"
  -l
  -j "\"${CMAKE_SOURCE_DIR}/scripts/dll_wrapper_type_mapper_cuda_runtime.json\""
  --cpp-file "\"${cuda_nvrtc_cpp_file}\""
  --header-file "\"${cuda_nvrtc_h_file}\""
  -up
  COMMENT "Generate CUDA NVRTC Wrappers"
  VERBATIM
)

# Set files as generated
set_source_files_properties(${cuda_runtime_cpp_file} ${cuda_runtime_h_file} ${cuda_nvrtc_cpp_file} ${cuda_nvrtc_h_file} PROPERTIES GENERATED TRUE)
source_group("Generated" FILES ${cuda_runtime_cpp_file} ${cuda_runtime_h_file} ${cuda_nvrtc_cpp_file} ${cuda_nvrtc_h_file})

# Create module for the CUDA wrappers
dmt_add_module_library(dmt-cuda-wrappers cuda-wrappers)

target_sources(dmt-cuda-wrappers
  PUBLIC
    FILE_SET dmt_cuda_wrappers_headers TYPE HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}
    FILES ${cuda_runtime_h_file} ${cuda_nvrtc_h_file}
  PRIVATE
    ${cuda_runtime_cpp_file} ${cuda_nvrtc_cpp_file}
)

# Add dependencies for copying DLLs before building the target
add_dependencies(dmt-cuda-wrappers copy_cuda_dlls copy_nvrtc_dlls)

# Link target to the platform
target_link_libraries(dmt-cuda-wrappers PUBLIC dmt-platform)

# Install generated headers
install(TARGETS dmt-cuda-wrappers FILE_SET dmt_cuda_wrappers_headers)

# Set the folder for Visual Studio to organize the target
set_target_properties(dmt-cuda-wrappers PROPERTIES FOLDER "Modules")
