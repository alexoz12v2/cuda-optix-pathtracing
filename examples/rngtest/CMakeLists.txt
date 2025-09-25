dmt_add_example(dmt-rngtest)
target_sources(dmt-rngtest PRIVATE main.cpp)

target_link_libraries(dmt-rngtest PRIVATE dmt-platform stb::stb)
