
add_executable(dmt-win32playground WIN32)
add_executable(dmt-win32playground-launcher)

# TODO refactor
set_target_properties(dmt-win32playground PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>)
set_target_properties(dmt-win32playground PROPERTIES DEBUG_POSTFIX -d)
set_target_properties(dmt-win32playground PROPERTIES FOLDER "Examples")
set_target_properties(dmt-win32playground PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>)
dmt_set_target_compiler_versions(dmt-win32playground)
dmt_set_target_warnings(dmt-win32playground)
dmt_set_target_optimization(dmt-win32playground)
dmt_add_compile_definitions(dmt-win32playground)


set_target_properties(dmt-win32playground-launcher PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>
  DEBUG_POSTFIX -d
  FOLDER "Examples"
  VS_DEBUGGER_WORKING_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>
  OUTPUT_NAME "dmt-win32playground"
  SUFFIX .com
)
dmt_set_target_compiler_versions(dmt-win32playground-launcher)
dmt_set_target_warnings(dmt-win32playground-launcher)
dmt_set_target_optimization(dmt-win32playground-launcher)
dmt_add_compile_definitions(dmt-win32playground-launcher)


target_sources(dmt-win32playground PRIVATE main.cpp)
target_sources(dmt-win32playground-launcher PRIVATE launcher.cpp)
target_link_libraries(dmt-win32playground PRIVATE dmt-platform-mixed)
add_dependencies(dmt-win32playground-launcher dmt-win32playground)
