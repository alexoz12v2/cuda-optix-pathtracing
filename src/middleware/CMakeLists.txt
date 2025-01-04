dmt_add_module_library(dmt-middleware middleware
  MODULE_INTERFACE middleware.cppm
  MODULE_IMPLEMENTATION middleware.cpp
  HEADERS middleware.h
)

target_link_libraries(dmt-middleware 
  PUBLIC
    dmt::platform
    glm::glm
)