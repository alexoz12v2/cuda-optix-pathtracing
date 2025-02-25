add_library(stb INTERFACE)
target_sources(stb
  INTERFACE FILE_SET stb_header TYPE HEADERS FILES stb_image.h
)

add_library(stb::stb ALIAS stb)
install(TARGETS stb FILE_SET stb_header)