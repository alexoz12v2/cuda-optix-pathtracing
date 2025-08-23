dmt_add_cli(dmt-tracer)
target_sources(dmt-tracer
  PRIVATE main.cpp CLManager.cpp
)
target_include_directories(dmt-tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-tracer PRIVATE dmt::platform dmt::core stb::stb OpenEXR::OpenEXR)
