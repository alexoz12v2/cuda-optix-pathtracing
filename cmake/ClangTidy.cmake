# From SFML's Repo
if (NOT EXISTS ${CLANG_TIDY_EXECUTABLE})
  set(hints "")
  if (DMT_OS_WINDOWS)
    list(APPEND hints "C:\\Program Files\\LLVM\\bin" "$ENV{USERPROFILE}")
  endif ()
  find_program(CLANG_TIDY_EXEC_TEMP ${CLANG_TIDY_EXECUTABLE}
    HINTS ${hints}
  )
  if (CLANG_TIDY_EXEC_TEMP)
    set(CLANG_TIDY_EXECUTABLE ${CLANG_TIDY_EXEC_TEMP})
    unset(CLANG_TIDY_EXEC_TEMP)
  else ()
    message(FATAL_ERROR "Unable to find clang-tidy executable: \"${CLANG_TIDY_EXEC_TEMP}\"")
  endif ()
endif ()

# check version
execute_process(COMMAND ${CLANG_TIDY_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_TIDY_VERSION)
string(REGEX MATCH "version ([0-9]+)" CLANG_TIDY_VERSION ${CLANG_TIDY_VERSION}) # subgroups in CMAKE_MATCH_n
unset(CLANG_TIDY_VERSION)

# second match, after the dot If you use a newer feature, remember to update this!
if (CMAKE_MATCH_1 GREATER_EQUAL 16)
  message(STATUS "Using clang-tidy version ${CMAKE_MATCH_1}")
else ()
  message(FATAL_ERROR "clang-tidy version ${CMAKE_MATCH_1} is too low")
endif ()

# find run-clang-tidy script and run it with python
if (NOT RUN_CLANG_TIDY)
  set(hints "")
  if (DMT_OS_WINDOWS)
    list(APPEND hints "C:\\Program Files\\LLVM\\bin" "$ENV{USERPROFILE}")
  endif ()
  find_package(Python 3 REQUIRED)
  find_program(RUN_CLANG_TIDY run-clang-tidy)
endif ()

# vroom vroom
message(STATUS "Executing run-clang-tidy on folder ${PROJECT_BINARY_DIR}, files ${FORMAT_SOURCES}")
# execute_process(COMMAND ${Python_EXECUTABLE} ${RUN_CLANG_TIDY} -clang-tidy-binary ${CLANG_TIDY_EXECUTABLE} -quiet
# -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy -p ${PROJECT_BINARY_DIR} RESULT_VARIABLE EXIT_CODE)
set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXECUTABLE}; -quiet; -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy; -p
  ${PROJECT_BINARY_DIR};)
