# possible todo: OS specific files, usage of target_precompile_headers

#dmt_add_module_library(dmt-testdmt testdmt)
#target_sources(dmt-testdmt
#  PUBLIC
#    FILE_SET CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/testdmt
#    FILES ${CMAKE_SOURCE_DIR}/include/testdmt/testImplementationUnit.cppm
#  PRIVATE
#    testImplementationUnit.cpp
#)
dmt_add_module_library(dmt-testdmt testdmt
  MODULE_INTERFACE testdmt.cppm
  MODULE_IMPLEMENTATION testdmt.cpp
)
