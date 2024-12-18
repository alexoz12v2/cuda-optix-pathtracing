dmt_add_module_library(dmt-platform platform
  MODULE_INTERFACE platform.cppm
  MODULE_IMPLEMENTATION platform.cpp
  MODULE_PARTITION_INTERFACES platform-logging.cppm platform-threadPool.cppm
  MODULE_PARTITION_IMPLEMENTATIONS platform-logging.cpp platform-threadPool.cpp
)
# target_link_libraries(dmt-platform PRIVATE fmt::fmt)
target_link_libraries(dmt-platform 
  PRIVATE 
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:Kernel32.lib> # needed by fileapi.h
    glad::glad 
    glfw
    imgui::imgui
) 