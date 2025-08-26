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

# ---------------------------------------------
# Helper function: copy one runtime (Debug/Release)
# Arguments:
#   runtime_var - CMake variable name (not value) holding path to runtime
#   dest_dir    - bin or lib
#   config      - "Release" or "Debug"
# ---------------------------------------------
function(fbxsdk_copy_runtime runtime_var dest_dir config)
    set(_src "${${runtime_var}}")
    if(NOT EXISTS "${_src}")
        message(FATAL_ERROR "Runtime ${runtime_var} not found at ${_src}")
    endif()

    # If already in build dir, skip copying
    string(FIND "${_src}" "${CMAKE_BINARY_DIR}" _in_build_dir)
    if(_in_build_dir EQUAL 0)
        message(STATUS "Skipping copy of ${runtime_var}, already in build dir: ${_src}")
        return()
    endif()

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${dest_dir}")

    # Extract filename + extension
    get_filename_component(_filename "${_src}" NAME)
    get_filename_component(_name_we "${_src}" NAME_WE)
    get_filename_component(_ext "${_src}" EXT)

    # For debug, add -d suffix before extension
    if(config STREQUAL "Debug")
        set(_dst "${CMAKE_BINARY_DIR}/${dest_dir}/${_name_we}-d${_ext}")
    else()
        set(_dst "${CMAKE_BINARY_DIR}/${dest_dir}/${_filename}")
    endif()

    # Copy
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "${_dst}"
        RESULT_VARIABLE _copy_result
    )

    if(_copy_result EQUAL 0)
        # Reassign variable to new copied path
        set(${runtime_var} "${_dst}" CACHE FILEPATH "FBXSDK runtime (${config})" FORCE)
    else()
        message(FATAL_ERROR "Failed to copy ${runtime_var} from ${_src} to ${_dst}")
    endif()
endfunction()


if (DMT_OS_WINDOWS)
    set(_fbxsdk_vstudio_version "vs2022")
    set(_fbxsdk_approot "C:/Program Files/Autodesk/FBX/FBX SDK/2020.3.7")
    set(_fbxsdk_libdir_debug "lib/x64/debug")
    set(_fbxsdk_libdir_release "lib/x64/release")

    # libs + dll names
    set(_fbxsdk_libname_debug "libfbxsdk.lib")
    set(_fbxsdk_libname_release "libfbxsdk.lib")
    #set(_fbxsdk_libname_debug "libfbxsdk-md.lib")
    #set(_fbxsdk_libname_release "libfbxsdk-md.lib")
    set(_fbxsdk_dll_name "libfbxsdk.dll")

    set(_fbxsdk_alembic_libname_debug "alembic-md.lib")
    set(_fbxsdk_alembic_libname_release "alembic-md.lib")
    set(_fbxsdk_libxml2_libname_debug "libxml2-md.lib")
    set(_fbxsdk_libxml2_libname_release "libxml2-md.lib")
    set(_fbxsdk_zlib_libname_debug "zlib-md.lib")
    set(_fbxsdk_zlib_libname_release "zlib-md.lib")

    set(_fbxsdk_root "${_fbxsdk_approot}")
    message("_fbxsdk_root: ${_fbxsdk_root}")

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
    message(STATUS "FBXSDK_LIBRARY ${FBXSDK_LIBRARY}")
    message(STATUS "FBXSDK_LIBRARY_DEBUG ${FBXSDK_LIBRARY_DEBUG}")

    find_file(FBXSDK_RUNTIME ${_fbxsdk_dll_name}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_file(FBXSDK_RUNTIME_DEBUG ${_fbxsdk_dll_name}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})
    message(STATUS "FBXSDK_RUNTIME ${FBXSDK_RUNTIME}")
    message(STATUS "FBXSDK_RUNTIME_DEBUG ${FBXSDK_RUNTIME_DEBUG}")

    # dependencies
    find_library(FBXSDK_ALEMBIC_LIBRARY ${_fbxsdk_alembic_libname_release}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_ALEMBIC_LIBRARY_DEBUG ${_fbxsdk_alembic_libname_debug}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_LIBXML2_LIBRARY ${_fbxsdk_libxml2_libname_release}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_LIBXML2_LIBRARY_DEBUG ${_fbxsdk_libxml2_libname_debug}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_ZLIB_LIBRARY ${_fbxsdk_zlib_libname_release}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_ZLIB_LIBRARY_DEBUG ${_fbxsdk_zlib_libname_debug}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    if (FBXSDK_INCLUDE_DIR AND FBXSDK_LIBRARY AND FBXSDK_LIBRARY_DEBUG AND FBXSDK_RUNTIME)
        set(FBXSDK_FOUND YES)
    else()
        set(FBXSDK_FOUND NO)
    endif()

    if (FBXSDK_FOUND)
        # copy DLLs into bin
        fbxsdk_copy_runtime(FBXSDK_RUNTIME "bin" "Release")
        fbxsdk_copy_runtime(FBXSDK_RUNTIME_DEBUG "bin" "Debug")

        add_library(FBXSDK::fbxsdk SHARED IMPORTED)
        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            IMPORTED_IMPLIB_RELEASE "${FBXSDK_LIBRARY}"
            IMPORTED_IMPLIB_DEBUG   "${FBXSDK_LIBRARY_DEBUG}"
            IMPORTED_LOCATION_RELEASE "${FBXSDK_RUNTIME}"
            IMPORTED_LOCATION_DEBUG   "${FBXSDK_RUNTIME_DEBUG}"
            INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
        )

        target_link_libraries(FBXSDK::fbxsdk INTERFACE
            "$<$<CONFIG:Release>:${FBXSDK_ALEMBIC_LIBRARY};${FBXSDK_LIBXML2_LIBRARY};${FBXSDK_ZLIB_LIBRARY}>"
            "$<$<CONFIG:Debug>:${FBXSDK_ALEMBIC_LIBRARY_DEBUG};${FBXSDK_LIBXML2_LIBRARY_DEBUG};${FBXSDK_ZLIB_LIBRARY_DEBUG}>"
        )

        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG Debug
            MAP_IMPORTED_CONFIG_RELEASE Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
        )
        message(STATUS "FBX SDK imported target created (Windows): FBXSDK::fbxsdk")
    endif()

elseif (DMT_OS_LINUX)
    if (NOT DEFINED ENV{FBX_SDK_PATH})
        message(FATAL_ERROR "FBX_SDK_PATH is not defined. Please set it to your FBX SDK root directory.")
    endif()

    set(_fbxsdk_approot $ENV{FBX_SDK_PATH})
    set(_fbxsdk_libdir_debug "lib/debug")
    set(_fbxsdk_libdir_release "lib/release")

    set(_fbxsdk_libname "libfbxsdk.a")
    set(_fbx_runtime_lib "libfbxsdk.so")
    set(_fbxsdk_alembic_libname "libalembic.a")

    set(_fbxsdk_root "${_fbxsdk_approot}")
    message("_fbxsdk_root: ${_fbxsdk_root}")

    find_path(FBXSDK_INCLUDE_DIR "fbxsdk.h"
              PATHS ${_fbxsdk_root}
              PATH_SUFFIXES "include")

    find_library(FBXSDK_LIBRARY ${_fbxsdk_libname}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_LIBRARY_DEBUG ${_fbxsdk_libname}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_RUNTIME ${_fbx_runtime_lib}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_RUNTIME_DEBUG ${_fbx_runtime_lib}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_ALEMBIC_LIBRARY ${_fbxsdk_alembic_libname}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_ALEMBIC_LIBRARY_DEBUG ${_fbxsdk_alembic_libname}
                 PATHS ${_fbxsdk_root} PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    if (FBXSDK_INCLUDE_DIR AND FBXSDK_LIBRARY AND FBXSDK_RUNTIME)
        set(FBXSDK_FOUND YES)
    else()
        set(FBXSDK_FOUND NO)
    endif()

    if (FBXSDK_FOUND)
        # copy SOs into lib
        fbxsdk_copy_runtime(FBXSDK_RUNTIME "lib" "Release")
        fbxsdk_copy_runtime(FBXSDK_RUNTIME_DEBUG "lib" "Debug")

        add_library(FBXSDK::fbxsdk SHARED IMPORTED)
        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            IMPORTED_LOCATION_RELEASE "${FBXSDK_RUNTIME}"
            IMPORTED_LOCATION_DEBUG   "${FBXSDK_RUNTIME_DEBUG}"
            INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
        )

        target_link_libraries(FBXSDK::fbxsdk INTERFACE
            "$<$<CONFIG:Release>:${FBXSDK_RUNTIME};${FBXSDK_ALEMBIC_LIBRARY}>"
            "$<$<CONFIG:Debug>:${FBXSDK_RUNTIME_DEBUG};${FBXSDK_ALEMBIC_LIBRARY_DEBUG}>"
        )

        set_target_properties(FBXSDK::fbxsdk PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG Debug
            MAP_IMPORTED_CONFIG_RELEASE Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
        )
        message(STATUS "FBX SDK imported target created (Linux): FBXSDK::fbxsdk")
    endif()
else()
    message(FATAL_ERROR "Unrecognized OS")
endif()
