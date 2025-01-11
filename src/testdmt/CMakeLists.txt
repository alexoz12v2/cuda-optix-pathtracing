# possible todo: OS specific files, usage of target_precompile_headers

#dmt_add_module_library(dmt-testdmt testdmt)
#target_sources(dmt-testdmt
#  PUBLIC
#    FILE_SET CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/testdmt
#    FILES ${CMAKE_SOURCE_DIR}/include/testdmt/testImplementationUnit.cppm
#  PRIVATE
#    testImplementationUnit.cpp
#)
dmt_add_module_library(dmt-testdmt testdmt)
target_sources(dmt-testdmt
  PUBLIC 
    FILE_SET testdmt TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/testdmt ${CMAKE_CURRENT_SOURCE_DIR}
      FILES ${CMAKE_SOURCE_DIR}/include/testdmt/testdmt.cppm ${CMAKE_CURRENT_SOURCE_DIR}/testdmt.cpp
)
