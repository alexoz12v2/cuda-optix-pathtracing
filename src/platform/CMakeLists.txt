# --- Targets ---
dmt_add_module_library(dmt-platform platform SHARED)

# --- Common headers/sources ---
set(HEADER_DIR "${CMAKE_SOURCE_DIR}/include/platform")

set(platform_headers
  ${HEADER_DIR}/platform-context.h
  ${HEADER_DIR}/platform-file.h
  ${HEADER_DIR}/platform-launch.h
  ${HEADER_DIR}/platform-logging.h
  ${HEADER_DIR}/platform-macros.h
  ${HEADER_DIR}/platform-memory.h
  ${HEADER_DIR}/platform-memory-stackAllocator.h
  ${HEADER_DIR}/platform-threadPool.h
  ${HEADER_DIR}/platform-utils.h
  ${HEADER_DIR}/platform.h
)

set(platform_private_sources
  platform-context.cpp
  platform-launch.cpp
  platform-logging.cpp
  platform-memory.cpp
  platform-memory-stackAllocator.cpp
  platform-threadPool.cpp
  platform-utils.cpp
  platform.cpp
)

# --- Per-platform sources/headers ---
set(PLATFORM_SOURCES_WIN32
  platform-file.win32.cpp
  platform-logging.win32.cpp
  platform-utils.win32.cpp
  platform-memory.win32.cpp
  platform-threadPool.win32.cpp
  platform-os-utils.win32.cpp
)
set(PLATFORM_HEADERS_WIN32 platform-os-utils.win32.h)

set(PLATFORM_SOURCES_LINUX
  platform-file.linux.cpp
  platform-logging.linux.cpp
  platform-memory.linux.cpp
  platform-threadPool.linux.cpp
  platform-utils.linux.cpp
  platform-os-utils.linux.cpp
)
set(PLATFORM_HEADERS_LINUX platform-os-utils.linux.h)

# --- Platform selection ---
if(DMT_OS_WINDOWS)
  set(platform_private_sources ${platform_private_sources} ${PLATFORM_SOURCES_WIN32})
  set(platform_private_headers ${PLATFORM_HEADERS_WIN32})
  set(platform_unused_sources ${PLATFORM_SOURCES_LINUX} ${PLATFORM_HEADERS_LINUX})
elseif(DMT_OS_LINUX)
  set(platform_private_sources ${platform_private_sources} ${PLATFORM_SOURCES_LINUX})
  set(platform_private_headers ${PLATFORM_HEADERS_LINUX})
  set(platform_unused_sources ${PLATFORM_SOURCES_WIN32} ${PLATFORM_HEADERS_WIN32})
endif()

# --- Mark unused files as headers-only so they appear in IDEs but are not built ---
set_source_files_properties(${platform_unused_sources} ${platform_private_headers} PROPERTIES HEADER_FILE_ONLY TRUE)

source_group("Unused" FILES ${platform_unused_sources})

# --- dmt-platform target ---
target_sources(dmt-platform
  PUBLIC
    FILE_SET platform_headers TYPE HEADERS BASE_DIRS ${HEADER_DIR}
      FILES ${platform_headers}
  PRIVATE
    ${platform_private_sources}
    ${platform_private_headers}
    ${platform_private_sources}
    ${platform_unused_sources}
)

target_link_libraries(dmt-platform
  PRIVATE
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread>
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll>
    backward
)

set_target_properties(dmt-platform PROPERTIES FOLDER "Modules/Platform")

# --- Install ---
install(TARGETS dmt-platform FILE_SET platform_headers)
