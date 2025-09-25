dmt_add_cli(dmt-test-cuda)
target_sources(dmt-test-cuda PRIVATE main.cpp cuda-test.cpp)
target_include_directories(dmt-test-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-test-cuda PRIVATE dmt::platform dmt::core stb::stb dmt::cuda-wrappers OpenEXR::OpenEXR)
