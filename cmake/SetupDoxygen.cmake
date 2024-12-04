
macro(dmt_setup_doxygen)
  find_package(Doxygen REQUIRED dot)
  if(NOT DOXYGEN_FOUND)
    message(FATAL_ERROR "doxygen executable is required to build documentation")
  else()
    message(STATUS "found ${DOXYGEN_EXECUTABLE} executable --version ${DOXYGEN_VERSION}")
  endif()

  set(DOXYGEN_GENERATE_HTML YES)
  doxygen_add_docs(dmt-doxygen
    ${PROJECT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "generate doxygen"
    CONFIG_FILE ${PROJECT_SOURCE_DIR}/docs/Doxyfile
  )
endmacro()