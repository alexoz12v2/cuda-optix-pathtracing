dmt_add_module_library(dmt-platform-os-utils platform SHARED)
dmt_add_module_library(dmt-platform platform SHARED)

string(CONCAT platform_private_sources 
    "platform-utils.cpp;"
    "platform-logging.cpp;"
    "platform-threadPool.cpp;"
    "platform-memory.cpp;"
)
string(CONCAT platform_win32_private_sources
  "platform-memory.win32.cpp;"
  "platform-logging.win32.cpp;"
  "platform-utils.win32.cpp;"
  "platform-file.win32.cpp"
)
string(CONCAT platfrom_linux_private_sources
  "platform-logging.linux.cpp;"
  "platform-memory.linux.cpp;"
  "platform-utils.linux.cpp;"
  "platform-file.linux.cpp"
)

string(CONCAT platform_win32_private_headers
    "platform-os-utils.win32.h"
)
string(CONCAT platform_linux_private_headers
    "platform-os-utils.linux.h"
)
set_source_files_properties(${platform_win32_private_headers} ${platform_linux_private_headers} PROPERTIES HEADER_FILE_ONLY TRUE)
set(platform_private_headers "")

string(CONCAT platform_os_private_sources_win32
  "platform-os-utils.win32.cpp"
)
string(CONCAT platform_os_private_sources_linux
  "platform-os-utils.linux.cpp"
)
set(platform_os_private_sources "")

# concat on private sources the ones for the current operating system
set(platform_unused_private_source "")
set(platform_os_unused_private_source "")
if(DEFINED DMT_OS_WINDOWS)
  list(APPEND platform_private_sources ${platform_win32_private_sources})
  list(APPEND platform_unused_private_source ${platfrom_linux_private_sources})
  list(APPEND platform_private_headers ${platform_win32_private_headers})
  list(APPEND platform_os_private_sources ${platform_os_private_sources_win32})
  list(APPEND platform_os_unused_private_source ${platform_os_private_sources_linux} ${platform_linux_private_headers})
elseif(DEFINED DMT_OS_LINUX)
  list(APPEND platform_private_sources ${platfrom_linux_private_sources})
  list(APPEND platform_unused_private_source ${platform_win32_private_sources})
  list(APPEND platform_private_headers ${platform_linux_private_headers})
  list(APPEND platform_os_private_sources ${platform_os_private_sources_linux})
  list(APPEND platform_os_unused_private_source ${platform_os_private_sources_win32} ${platform_win32_private_headers})
endif()
set_source_files_properties(${platform_os_unused_private_source} PROPERTIES HEADER_FILE_ONLY TRUE)

target_sources(dmt-platform-os-utils
  PUBLIC
    FILE_SET platform_os_utils_headers TYPE HEADERS FILES
      platform-os-utils-macros.h
      ${platform_private_headers}
  PRIVATE
    ${platform_os_private_sources}
  PRIVATE
    ${platform_os_unused_private_source}
)
source_group("Unused" FILES ${platform_os_unused_private_source})

target_sources(dmt-platform
  PUBLIC
    FILE_SET platform_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-logging.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-threadPool.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-memory.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-macros.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform-context.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform-file.h
  PRIVATE 
    ${platform_private_sources}
  PRIVATE # display only files
    ${platform_unused_private_source}
)

# display unused sources anyways
set_source_files_properties(${platform_unused_private_source} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("Unused" FILES ${platform_unused_private_source})

target_link_libraries(dmt-platform 
  PUBLIC
    glad 
    glfw 
    imgui 
  PRIVATE 
    dmt-platform-os-utils
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
) 

dmt_add_module_library(dmt-platform-cuda platform)
string(CONCAT platform_cuda_private_sources
  "platform-cuda-utils.cu;"
  "cudaTest.cu;"
  "platform-context.cu;" # keep ; as we will concatenate more
)
string(CONCAT platform_cuda_win32_sources
  "platform-cuda-fileMapping.win32.cu"
)
string(CONCAT platform_cuda_linux_sources
  "platform-cuda-fileMapping.linux.cu"
)
set(platform_cuda_unused_private_sources "")
if(DEFINED DMT_OS_WINDOWS)
  list(APPEND platform_cuda_private_sources ${platform_cuda_win32_sources})
  list(APPEND platform_cuda_unused_private_sources ${platform_cuda_linux_sources})
elseif(DEFINED DMT_OS_LINUX)
  list(APPEND platform_cuda_private_sources ${platform_cuda_linux_sources})
  list(APPEND platform_cuda_unused_private_sources ${platform_cuda_win32_sources})
endif()
set_source_files_properties(${platform_cuda_unused_private_sources} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("Unused" FILES ${platform_cuda_unused_private_sources})

target_sources(dmt-platform-cuda
  PUBLIC
    FILE_SET platform_cuda_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.cuh
            ${CMAKE_SOURCE_DIR}/include/platform/cudaTest.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform-context.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-fileMapping.h
  PRIVATE
    ${platform_cuda_private_sources}
  PRIVATE # display only files
    ${platform_cuda_unused_private_sources}
)

target_link_libraries(dmt-platform-cuda 
  PUBLIC 
    dmt-platform
    dmtcuda-Optix8
  PRIVATE
    dmt-platform-os-utils
)

dmt_add_module_library(dmt-platform-mixed platform)
target_sources(dmt-platform-mixed
  PUBLIC
    FILE_SET platform_mized_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-display.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-mixed-macros.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform-launch.h
  PRIVATE
    platform-display.cpp platform.cpp
    platform-launch.cpp
)
target_link_libraries(dmt-platform-mixed 
  PUBLIC 
    dmt-platform-cuda
    dmt-platform
  PRIVATE
    dmt-platform-os-utils
)

set_target_properties(dmt-platform-os-utils dmt-platform-mixed dmt-platform dmt-platform-cuda PROPERTIES FOLDER "Modules/Platform")

install(TARGETS dmt-platform-os-utils dmt-platform dmt-platform-cuda dmt-platform-mixed
  FILE_SET platform_os_utils_headers
  FILE_SET platform_headers 
  FILE_SET platform_cuda_headers 
  FILE_SET platform_mized_headers
)
