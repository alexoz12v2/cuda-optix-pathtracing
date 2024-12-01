# possible todo: OS specific files, usage of target_precompile_headers

#dmt_add_module_library(dmt-projtest testdmt)
#target_sources(dmt-projtest
#  PUBLIC
#    FILE_SET CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/projtest
#    FILES ${CMAKE_SOURCE_DIR}/include/projtest/testImplementationUnit.cppm
#  PRIVATE
#    testImplementationUnit.cpp
#)
dmt_add_module_library(dmt-projtest testdmt
  MODULE_INTERAFACE testdmt.cppm
  MODULE_IMPLEMENTATION testdmt.cpp
)
