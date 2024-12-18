dmt_add_example(dmt-testbed
  PRIVATE_SOURCES main.cpp
  PRIVATE_DEPS dmt::platform glad glfw imgui 
)

target_include_directories(
  dmt-testbed 
  PRIVATE ${PROJECT_SOURCE_DIR}/extern/imgui
)