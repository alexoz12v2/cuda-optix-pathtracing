add_library(stb INTERFACE)
target_sources(stb
  INTERFACE FILE_SET stb_header TYPE HEADERS FILES stb_image.h stb_image_write.h
)

add_library(stb::stb ALIAS stb)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS stb FILE_SET stb_header)