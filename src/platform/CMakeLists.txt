dmt_add_module_library(dmt-platform platform SHARED)

string(CONCAT platform_private_sources 
    "platform-os-utils.h;"
    "platform-utils.cpp;"
    "platform-logging.cpp;"
    "platform-threadPool.cpp;"
    "platform-memory.cpp;"
    "platform-os-utils.cpp;"
)
string(CONCAT platform_win32_private_sources
  "platform-memory.win32.cpp;"
  "platform-logging.win32.cpp"
)
string(CONCAT platfrom_linux_private_sources
  "platform-logging.linux.cpp;"
  "platform-memory.linux.cpp"
)

# concat on private sources the ones for the current operating system
set(platform_unused_private_source "")
if(DEFINED DMT_OS_WINDOWS)
  list(APPEND platform_private_sources ${platform_win32_private_sources})
  list(APPEND platform_unused_private_source ${platfrom_linux_private_sources})
elseif(DEFINED DMT_OS_LINUX)
  list(APPEND platform_private_sources ${platfrom_linux_private_sources})
  list(APPEND platform_unused_private_source ${platform_win32_private_sources})
endif()

target_sources(dmt-platform
  PUBLIC
    FILE_SET platform_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-logging.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-threadPool.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-memory.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-macros.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform-context.h
  PRIVATE 
    ${platform_private_sources}
  PRIVATE # display only files
    ${platform_unused_private_source}
)

# display unused sources anyways
foreach(unused_source ${platform_unused_private_source})
  set_source_files_properties(${unused_source} PROPERTIES HEADER_FILE_ONLY TRUE)
endforeach()
source_group("Unused" FILES ${platform_unused_private_source})

target_link_libraries(dmt-platform 
  PUBLIC
    glad 
    glfw 
    imgui 
  PRIVATE 
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
) 

dmt_add_module_library(dmt-platform-cuda platform)
target_sources(dmt-platform-cuda
  PUBLIC
    FILE_SET platform_cuda_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.cuh
            ${CMAKE_SOURCE_DIR}/include/platform/cudaTest.h
			${CMAKE_SOURCE_DIR}/include/platform/platform-context.h
  PRIVATE
    platform-cuda-utils.cu cudaTest.cu platform-context.cu
)
target_link_libraries(dmt-platform-cuda 
  PUBLIC 
    dmt-platform
    dmtcuda-Optix8
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
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
)

set_target_properties(dmt-platform-mixed dmt-platform dmt-platform-cuda PROPERTIES FOLDER "Modules/Platform")

install(TARGETS dmt-platform dmt-platform-cuda dmt-platform-mixed
  FILE_SET platform_headers 
  FILE_SET platform_cuda_headers 
  FILE_SET platform_mized_headers
)
