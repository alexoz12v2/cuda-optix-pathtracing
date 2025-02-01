add_library(qoi INTERFACE)
target_sources(qoi
  INTERFACE FILE_SET qoi_header TYPE HEADERS FILES qoi.h
)

add_library(qoi::qoi ALIAS qoi)
install(TARGETS qoi FILE_SET qoi_header)