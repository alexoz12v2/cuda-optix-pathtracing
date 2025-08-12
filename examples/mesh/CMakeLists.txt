dmt_add_cli(dmt-mesh)
target_sources(dmt-mesh
  PRIVATE main.cpp
)
target_include_directories(dmt-mesh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-mesh PRIVATE dmt::core dmt::platform stb::stb OpenEXR::OpenEXR)
