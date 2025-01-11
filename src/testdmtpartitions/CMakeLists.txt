dmt_add_module_library(dmt-testdmtpartitions testdmtpartitions)
target_sources(dmt-testdmtpartitions
  PUBLIC
  FILE_SET testdmtpartitions TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/testdmtpartitions ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${CMAKE_SOURCE_DIR}/include/testdmtpartitions/testdmtpartitions.cppm
          ${CMAKE_SOURCE_DIR}/include/testdmtpartitions/testdmtpartitions-moretrig.cppm 
          ${CMAKE_SOURCE_DIR}/include/testdmtpartitions/testdmtpartitions-trig.cppm 
          ${CMAKE_SOURCE_DIR}/include/testdmtpartitions/testdmtpartitions-implonly.cppm
          ${CMAKE_CURRENT_SOURCE_DIR}/testdmtpartitions.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/testdmtpartitions-moretrig.cpp 
          ${CMAKE_CURRENT_SOURCE_DIR}/testdmtpartitions-trig.cpp 
          ${CMAKE_CURRENT_SOURCE_DIR}/testdmtpartitions-implonly.cpp
)