dmt_add_module_library(dmt-platform-os-utils platform SHARED)
dmt_add_module_library(dmt-platform platform SHARED)

block()
  set(HEADER_DIR "${CMAKE_SOURCE_DIR}/include/platform")
  set(platform_headers
      ${HEADER_DIR}/platform-context.h
      ${HEADER_DIR}/platform-display.h
      ${HEADER_DIR}/platform-file.h
      ${HEADER_DIR}/platform-launch.h
      ${HEADER_DIR}/platform-logging-default-formatters.h
      ${HEADER_DIR}/platform-logging.h
      ${HEADER_DIR}/platform-macros.h
      ${HEADER_DIR}/platform-memory.h
      ${HEADER_DIR}/platform-threadPool.h
      ${HEADER_DIR}/platform-utils.h
      ${HEADER_DIR}/platform.h
  )

  set(platform_private_sources
      platform-context.cpp
      platform-display.cpp
      platform-launch.cpp
      platform-logging-default-formatters.cpp
      platform-logging.cpp
      platform-memory.cpp
      platform-threadPool.cpp
      platform-utils.cpp
      platform.cpp
  )
  set(platform_win32_private_sources
      platform-file.win32.cpp
      platform-logging.win32.cpp
      platform-memory.win32.cpp
      platform-utils.win32.cpp
  )
  set(platform_linux_private_sources
      platform-file.linux.cpp
      platform-logging.linux.cpp
      platform-memory.linux.cpp
      platform-utils.linux.cpp
  )

  set(platform_win32_private_headers
    platform-os-utils.win32.h
  )
  set(platform_linux_private_headers
    platform-os-utils.linux.h
  )

  set_source_files_properties(${platform_win32_private_headers} ${platform_linux_private_headers} PROPERTIES HEADER_FILE_ONLY TRUE)
  set(platform_private_headers "")

  set(platform_os_private_sources_win32
      platform-os-utils.win32.cpp
  )
  set(platform_os_private_sources_linux
      platform-os-utils.linux.cpp
  )
  set(platform_os_private_sources "")

  # concat on private sources the ones for the current operating system
  set(platform_unused_private_source "")
  set(platform_os_unused_private_source "")

  if(DEFINED DMT_OS_WINDOWS)
    list(APPEND platform_private_sources ${platform_win32_private_sources})
    list(APPEND platform_unused_private_source ${platform_linux_private_sources})
    list(APPEND platform_private_headers ${platform_win32_private_headers})
    list(APPEND platform_os_private_sources ${platform_os_private_sources_win32})
    list(APPEND platform_os_unused_private_source ${platform_os_private_sources_linux} ${platform_linux_private_headers})
  elseif(DEFINED DMT_OS_LINUX)
    list(APPEND platform_private_sources ${platform_linux_private_sources})
    list(APPEND platform_unused_private_source ${platform_win32_private_sources})
    list(APPEND platform_private_headers ${platform_linux_private_headers})
    list(APPEND platform_os_private_sources ${platform_os_private_sources_linux})
    list(APPEND platform_os_unused_private_source ${platform_os_private_sources_win32} ${platform_win32_private_headers})
  endif()

  set_source_files_properties(${platform_os_unused_private_source} PROPERTIES HEADER_FILE_ONLY TRUE)

  target_sources(dmt-platform-os-utils
    PUBLIC
      FILE_SET platform_os_utils_headers TYPE HEADERS FILES platform-os-utils-macros.h ${platform_private_headers}
    PRIVATE
      ${platform_os_private_sources}
      ${platform_os_unused_private_source}
  )
  source_group("Unused" FILES ${platform_os_unused_private_source})

  target_sources(dmt-platform
    PUBLIC
      FILE_SET platform_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform FILES ${platform_headers}
    PRIVATE
      ${platform_private_sources}
      ${platform_unused_private_source}
  )

  # display unused sources anyways
  set_source_files_properties(${platform_unused_private_source} PROPERTIES HEADER_FILE_ONLY TRUE)
  source_group("Unused" FILES ${platform_unused_private_source})
endblock()

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
    nlohmann_json::nlohmann_json
)

set_target_properties(dmt-platform-os-utils dmt-platform PROPERTIES FOLDER "Modules/Platform")

install(TARGETS dmt-platform-os-utils dmt-platform
  FILE_SET platform_os_utils_headers
  FILE_SET platform_headers
)
