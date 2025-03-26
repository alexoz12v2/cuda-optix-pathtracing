dmt_add_example(dmt-win32playground)
target_sources(dmt-win32playground
  PRIVATE main.cpp
)
target_link_libraries(dmt-win32playground PRIVATE dmt-platform)
