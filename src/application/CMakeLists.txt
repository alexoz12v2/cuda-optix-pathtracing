dmt_add_module_library(dmt-application application SHARED)

set(HEADER_DIR "${CMAKE_SOURCE_DIR}/include/application")
set(application_headers ${HEADER_DIR}/application-macros.h ${HEADER_DIR}/application-display.h)
set(application_sources application-display.cpp)

target_sources(
  dmt-application
  PUBLIC FILE_SET
         application_header_set
         TYPE
         HEADERS
         BASE_DIRS
         ${HEADER_DIR}
         FILES
         ${application_headers}
  PRIVATE ${application_sources})
set_target_properties(dmt-application PROPERTIES FOLDER "Modules")

target_link_libraries(dmt-application PUBLIC dmt::platform glfw imgui::imgui dmt::cuda-wrappers)
