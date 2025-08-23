#################################################################################
# CUDA Driver
#################################################################################

# generate library files  
set(cuda_driver_cpp_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-wrappers-cuda-driver.cpp)
set(cuda_driver_h_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-wrappers-cuda-driver.h)
set(cuda_driver_files ${cuda_driver_cpp_file} ${cuda_driver_h_file})

if(DMT_OS_LINUX)
  execute_process(
      OUTPUT_VARIABLE CUDA_LINUX_LIB_PATH
      COMMAND /bin/sh -c "ldconfig -p | grep 'libcuda.so ' | awk '{print $4}'"
      COMMENT "Finding libcuda.so path on Linux"
  )
  string(STRIP "${CUDA_LINUX_LIB_PATH}" CUDA_LINUX_LIB_PATH) # Remove any trailing newline
else()
  set(CUDA_LINUX_LIB_PATH "")
endif()

add_custom_command(
  OUTPUT ${cuda_driver_files}
  COMMAND ${PYTHON_EXEC} 
  ${CMAKE_SOURCE_DIR}/scripts/generate_dll_wrapper_file.py 
  "{ \"Windows\": \"C:\\\\Windows\\\\System32\\\\nvcuda.dll\", \"Linux\": \"${CUDA_LINUX_LIB_PATH}\" }"
  --name CUDADriverLibrary
  -i cuda.h 
  -mf "_v2"
  -v "_v{n}" 
  -j ${CMAKE_SOURCE_DIR}/scripts/dll_wrapper_type_mapper_cuda_driver.json 
  --cpp-file ${cuda_driver_cpp_file}
  --header-file ${cuda_driver_h_file}
  -up
  -em DMT_PLATFORM_API
  COMMENT "Python Script to generate library wrappers for the CUDA Driver API"
  VERBATIM
)

#################################################################################
# CUDA Runtime
#################################################################################

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

# get file names
get_filename_component(cudart_name ${cudart_path} NAME)

# declare out shared lib files
set(cudart_build_path ${CMAKE_BINARY_DIR}/bin/${cudart_name})

# Copy CUDA DLLs using custom targets
add_custom_command(
  OUTPUT ${cudart_build_path}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${cudart_path} ${CMAKE_BINARY_DIR}/bin
  COMMENT "Copying cudart.dll to the bin directory"
)
add_custom_target(copy_cuda_dlls ALL DEPENDS ${cudart_build_path})

# Ensure these copy steps run after the build
add_dependencies(copy_cuda_dlls dmt-platform)

# Define the paths for the generated files
set(cuda_runtime_cpp_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-wrappers-cuda-runtime.cpp)
set(cuda_runtime_h_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-wrappers-cuda-runtime.h)

# Custom commands for generating wrapper files
add_custom_command(
  OUTPUT ${cuda_runtime_cpp_file} ${cuda_runtime_h_file}
  COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/scripts/generate_dll_wrapper_file.py
  "{ \"Windows\": \"${cudart_build_path}\", \"Linux\": \"${cudart_build_path}\" }"
  --name CUDARTLibrary
  -i cuda_runtime_api.h glad/gl.h cuda_gl_interop.h
  -v "_v{n}"
  -l
  -j "\"${CMAKE_SOURCE_DIR}/scripts/dll_wrapper_type_mapper_cuda_runtime.json\""
  --cpp-file "\"${cuda_runtime_cpp_file}\""
  --header-file "\"${cuda_runtime_h_file}\""
  -up
  -pexe
  COMMENT "Generate CUDA Runtime API Wrappers"
  VERBATIM
)
add_custom_target(copy_nvidia_driver ALL DEPENDS ${cuda_runtime_cpp_file} ${cuda_runtime_h_file})
add_dependencies(copy_nvidia_driver dmt-platform)

#################################################################################
# CUDA NVRTC
#################################################################################

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

# get file names
get_filename_component(nvrtc_name ${nvrtc_path} NAME)

# declare out shared lib files
set(nvrtc_build_path ${CMAKE_BINARY_DIR}/bin/${nvrtc_name})

# Copy CUDA DLLs using custom targets
add_custom_command(
  OUTPUT ${nvrtc_build_path}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${nvrtc_path} ${CMAKE_BINARY_DIR}/bin
  COMMENT "Copying nvrtc.dll to the bin directory"
)
add_custom_target(copy_nvrtc_dlls ALL DEPENDS ${nvrtc_build_path})

# Ensure these copy steps run after the build
add_dependencies(copy_nvrtc_dlls dmt-platform)

# Define the paths for the generated files
set(cuda_nvrtc_cpp_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-wrappers-nvrtc.cpp)
set(cuda_nvrtc_h_file ${CMAKE_CURRENT_BINARY_DIR}/cuda-wrappers/cuda-wrappers-nvrtc.h)

# Custom commands for generating wrapper files
add_custom_command(
  OUTPUT ${cuda_nvrtc_cpp_file} ${cuda_nvrtc_h_file}
  COMMAND ${PYTHON_EXEC} ${CMAKE_SOURCE_DIR}/scripts/generate_dll_wrapper_file.py
  "{ \"Windows\": \"${nvrtc_build_path}\", \"Linux\": \"${nvrtc_build_path}\" }"
  --name NVRTCLibrary
  -i nvrtc.h
  -v "_v{n}"
  -l
  -j "\"${CMAKE_SOURCE_DIR}/scripts/dll_wrapper_type_mapper_cuda_nvrtc.json\""
  --cpp-file "\"${cuda_nvrtc_cpp_file}\""
  --header-file "\"${cuda_nvrtc_h_file}\""
  -up
  -pexe
  COMMENT "Generate CUDA NVRTC Wrappers"
  VERBATIM
)

#################################################################################
# The rest
#################################################################################

# Set files as generated
set_source_files_properties(
  ${cuda_driver_files} ${cuda_runtime_cpp_file} ${cuda_runtime_h_file} ${cuda_nvrtc_cpp_file} 
    ${cuda_nvrtc_h_file} ${cudart_build_path} ${nvrtc_build_path} 
  PROPERTIES GENERATED TRUE)

source_group("Generated" FILES 
  ${cuda_driver_files} ${cuda_runtime_cpp_file} ${cuda_runtime_h_file} ${cuda_nvrtc_cpp_file} ${cuda_nvrtc_h_file})

# Create module for the CUDA wrappers
# NOTE: This is static
dmt_add_module_library(dmt-cuda-wrappers cuda-wrappers)

target_sources(dmt-cuda-wrappers
  PUBLIC
    FILE_SET dmt_cuda_wrappers_headers TYPE HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/include/cuda-wrappers FILES 
      ${cuda_runtime_h_file} ${cuda_nvrtc_h_file} ${cuda_driver_h_file}
      ${CMAKE_SOURCE_DIR}/include/cuda-wrappers/cuda-wrappers-utils.h
      ${CMAKE_SOURCE_DIR}/include/cuda-wrappers/cuda-wrappers-macros.h
  PRIVATE
    ${cuda_runtime_cpp_file} ${cuda_nvrtc_cpp_file} ${cuda_driver_cpp_file}
)

# Add dependencies for copying DLLs before building the target
add_dependencies(dmt-cuda-wrappers copy_cuda_dlls copy_nvrtc_dlls copy_nvidia_driver)

string(STRIP "$ENV{CUDA_HOME}" cuda_toolkit_headers)
if (NOT DEFINED ENV{CUDA_HOME})
  message(FATAL_ERROR "define CUDA_HOME")
endif()
set(cuda_toolkit_headers "${cuda_toolkit_headers}/include")

target_include_directories(dmt-cuda-wrappers PUBLIC "${cuda_toolkit_headers}")

# Link target to the platform
target_link_libraries(dmt-cuda-wrappers PUBLIC dmt-platform glad::glad)

# Install generated headers
install(TARGETS dmt-cuda-wrappers FILE_SET dmt_cuda_wrappers_headers)

# Set the folder for Visual Studio to organize the target
set_target_properties(dmt-cuda-wrappers copy_cuda_dlls copy_nvrtc_dlls PROPERTIES FOLDER "Modules/CUDA Wrappers")
