dmt_add_module_library(dmt-platform platform
  MODULE_INTERFACE platform.cppm
  MODULE_IMPLEMENTATION platform.cpp
  MODULE_PARTITION_INTERFACES platform-logging.cppm platform-memory.cppm platform-utils.cppm
  MODULE_PARTITION_IMPLEMENTATIONS platform-logging.cpp platform-memory.cpp platform-utils.cpp
)
# target_link_libraries(dmt-platform PRIVATE fmt::fmt)
target_link_libraries(dmt-platform 
  PRIVATE 
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
) 