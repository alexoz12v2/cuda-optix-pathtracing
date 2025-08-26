#
# Helper function for finding the FBX SDK.
#
# sets: FBXSDK_FOUND, 
#       FBXSDK_DIR, 
#       FBXSDK_LIBRARY, 
#       FBXSDK_LIBRARY_DEBUG
#       FBXSDK_INCLUDE_DIR
#
message("Looking for FBX SDK")

if (DMT_OS_WINDOWS)
    set(_fbxsdk_vstudio_version "vs2022")
    # Autodesk FBX SDK installation
    set(_fbxsdk_approot "C:/Program Files/Autodesk/FBX/FBX SDK/2020.3.7")
    set(_fbxsdk_libdir_debug "lib/x64/debug")
    set(_fbxsdk_libdir_release "lib/x64/release")

    # main SDK libs
    set(_fbxsdk_libname_debug "libfbxsdk-md.lib")
    set(_fbxsdk_libname_release "libfbxsdk-md.lib")

    # dependencies
    set(_fbxsdk_alembic_libname_debug "alembic-md.lib")
    set(_fbxsdk_alembic_libname_release "alembic-md.lib")
    set(_fbxsdk_libxml2_libname_debug "libxml2-md.lib")
    set(_fbxsdk_libxml2_libname_release "libxml2-md.lib")
    set(_fbxsdk_zlib_libname_debug "zlib-md.lib")
    set(_fbxsdk_zlib_libname_release "zlib-md.lib")

    # FBX SDK root
    set(_fbxsdk_root "${_fbxsdk_approot}")
    message("_fbxsdk_root: ${_fbxsdk_root}")

    # include
    find_path(FBXSDK_INCLUDE_DIR "fbxsdk.h"
              PATHS ${_fbxsdk_root}
              PATH_SUFFIXES "include")

    # main fbxsdk libs
    
    find_library(FBXSDK_LIBRARY ${_fbxsdk_libname_release}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_LIBRARY_DEBUG ${_fbxsdk_libname_debug}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    # dependency libs (just variables, not targets)
    find_library(FBXSDK_ALEMBIC_LIBRARY ${_fbxsdk_alembic_libname_release}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_ALEMBIC_LIBRARY_DEBUG ${_fbxsdk_alembic_libname_debug}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_LIBXML2_LIBRARY ${_fbxsdk_libxml2_libname_release}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_LIBXML2_LIBRARY_DEBUG ${_fbxsdk_libxml2_libname_debug}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_ZLIB_LIBRARY ${_fbxsdk_zlib_libname_release}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_ZLIB_LIBRARY_DEBUG ${_fbxsdk_zlib_libname_debug}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    if (FBXSDK_INCLUDE_DIR AND FBXSDK_LIBRARY AND FBXSDK_LIBRARY_DEBUG)
        set(FBXSDK_FOUND YES)
    else()
        set(FBXSDK_FOUND NO)
    endif()

    if (FBXSDK_FOUND)
        add_library(FBXSDK::fbxsdk STATIC IMPORTED)
        
        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            IMPORTED_LOCATION_RELEASE "${FBXSDK_LIBRARY}"
            IMPORTED_LOCATION_DEBUG   "${FBXSDK_LIBRARY_DEBUG}"
            INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
        )

        # Link hidden dependency libs into the fbxsdk target
        target_link_libraries(FBXSDK::fbxsdk 
            INTERFACE
                "$<$<CONFIG:Release>:${FBXSDK_ALEMBIC_LIBRARY};${FBXSDK_LIBXML2_LIBRARY};${FBXSDK_ZLIB_LIBRARY}>"
                "$<$<CONFIG:Debug>:${FBXSDK_ALEMBIC_LIBRARY_DEBUG};${FBXSDK_LIBXML2_LIBRARY_DEBUG};${FBXSDK_ZLIB_LIBRARY_DEBUG}>"
        )

        # Config mapping
        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG   Debug
            MAP_IMPORTED_CONFIG_RELEASE Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
        )

        message(STATUS "FBX SDK imported target created: FBXSDK::fbxsdk")
    endif()
elseif (DMT_OS_LINUX)
    if (NOT DEFINED ENV{FBX_SDK_PATH})
        message(FATAL_ERROR "FBX_SDK_PATH is not defined. Please set it to your FBX SDK root directory.")
    endif()

    set(_fbxsdk_approot $ENV{FBX_SDK_PATH})
    set(_fbxsdk_libdir_debug "lib/debug")
    set(_fbxsdk_libdir_release "lib/release")

    # main SDK libs
    set(_fbxsdk_libname "libfbxsdk.a")   # static archive (same for debug/release)
    set(_fbx_runtime_lib "libfbxsdk.so") # shared runtime
    set(_fbxsdk_alembic_libname "libalembic.a")

    # FBX SDK root
    set(_fbxsdk_root "${_fbxsdk_approot}")
    message("_fbxsdk_root: ${_fbxsdk_root}")

    # include
    find_path(FBXSDK_INCLUDE_DIR "fbxsdk.h"
              PATHS ${_fbxsdk_root}
              PATH_SUFFIXES "include")

    # main static library
    find_library(FBXSDK_LIBRARY ${_fbxsdk_libname}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_LIBRARY_DEBUG ${_fbxsdk_libname}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})
    find_library(FBXSDK_RUNTIME ${_fbx_runtime_lib}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_RUNTIME_DEBUG ${_fbx_runtime_lib}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    # alembic dependency
    find_library(FBXSDK_ALEMBIC_LIBRARY ${_fbxsdk_alembic_libname}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_ALEMBIC_LIBRARY_DEBUG ${_fbxsdk_alembic_libname}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    if (FBXSDK_INCLUDE_DIR AND FBXSDK_LIBRARY AND FBXSDK_RUNTIME)
        set(FBXSDK_FOUND YES)
    else()
        set(FBXSDK_FOUND NO)
        message(WARNING "FBX SDK Something was not found: FBXSDK_INCLUDE_DIR: ${FBXSDK_INCLUDE_DIR}, FBXSDK_LIBRARY: ${FBXSDK_LIBRARY}, FBXSDK_RUNTIME: ${FBXSDK_RUNTIME}")
    endif()

    if (FBXSDK_FOUND)
        add_library(FBXSDK::fbxsdk SHARED IMPORTED)

        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            IMPORTED_LOCATION_RELEASE "${FBXSDK_RUNTIME}"
            IMPORTED_LOCATION_DEBUG   "${FBXSDK_RUNTIME_DEBUG}"
            INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
        )

        # link hidden deps
        message(STATUS "FBXSDK_RUNTIME:FBXSDK_ALEMBIC_LIBRARY = ${FBXSDK_RUNTIME};${FBXSDK_ALEMBIC_LIBRARY}")
        message(STATUS "FBXSDK_RUNTIME_DEBUG:FBXSDK_ALEMBIC_LIBRARY_DEBUG = ${FBXSDK_RUNTIME_DEBUG};${FBXSDK_ALEMBIC_LIBRARY_DEBUG}")
        target_link_libraries(FBXSDK::fbxsdk 
          INTERFACE
            "$<$<CONFIG:Release>:${FBXSDK_RUNTIME};${FBXSDK_ALEMBIC_LIBRARY}>"
            "$<$<CONFIG:Debug>:${FBXSDK_RUNTIME_DEBUG};${FBXSDK_ALEMBIC_LIBRARY_DEBUG}>"
        )
        # config mapping
        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG   Debug
            MAP_IMPORTED_CONFIG_RELEASE Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
        )

        message(STATUS "FBX SDK imported target created (Linux): FBXSDK::fbxsdk")
    endif()
else()
    message(FATAL_ERROR "Unrecognized OS")
endif()

