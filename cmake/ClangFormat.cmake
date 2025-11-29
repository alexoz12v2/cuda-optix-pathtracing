# From SFML's Repo
if (NOT EXISTS ${CLANG_FORMAT_EXECUTABLE})
  set(hints "")
  if (DMT_OS_WINDOWS)
    list(APPEND hints "C:\\Program Files\\LLVM\\bin" "$ENV{USERPROFILE}")
  endif ()
  find_program(CLANG_FORMAT_EXEC_TEMP ${CLANG_FORMAT_EXECUTABLE}
    HINTS ${hints}
  )
  if (CLANG_FORMAT_EXEC_TEMP)
    set(CLANG_FORMAT_EXECUTABLE ${CLANG_FORMAT_EXEC_TEMP})
    unset(CLANG_FORMAT_EXEC_TEMP)
  else ()
    message(FATAL_ERROR "Unable to find clang-format executable: \"${CLANG_FORMAT_EXECUTABLE}\"")
  endif ()
endif ()

# check version
execute_process(COMMAND ${CLANG_FORMAT_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_FORMAT_VERSION)
string(REGEX MATCH "clang-format version ([0-9]+)" CLANG_FORMAT_VERSION ${CLANG_FORMAT_VERSION})
unset(CLANG_FORMAT_VERSION)

# second match, after the dot If you use a newer feature, remember to update this!
if (CMAKE_MATCH_1 GREATER_EQUAL 16)
  message(STATUS "Using clang-format version ${CMAKE_MATCH_1}")
else ()
  message(FATAL_ERROR "clang-format version ${CMAKE_MATCH_1} is too low")
endif ()

# fun for each file recognized as C++/CUDA source Update as many times as you need
set(FORMAT_SOURCES "")
foreach (FOLDER IN ITEMS include src test benchmark examples)
  file(
    GLOB_RECURSE
    folder_files
    "${FOLDER}/*.h"
    "${FOLDER}/*.hpp"
    "${FOLDER}/*.c"
    "${FOLDER}/*.cpp"
    "${FOLDER}/*.cc"
    "${FOLDER}/*.cu"
    "${FOLDER}/*.cuh"
    "${FOLDER}/*.ixx"
    "${FOLDER}/*.cppm")
  # Exclude Third Party code if you insert in the aforementioned folders! list(FILTER folder_files EXCLUDE REGEX)
  list(APPEND FORMAT_SOURCES ${folder_files})
endforeach ()

execute_process(COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${FORMAT_SOURCES})
add_custom_target(
  dmt-clang-format
  COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${FORMAT_SOURCES}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
