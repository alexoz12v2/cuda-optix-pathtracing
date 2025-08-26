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
# Helper macro: copy runtimes and reassign vars
# ---------------------------------------------
function(fbxsdk_copy_runtime runtime_var runtime_dbg_var dest_dir)
    # ensure dir exists
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${dest_dir}")

    # get basenames
    get_filename_component(_runtime_name "${${runtime_var}}" NAME)
    get_filename_component(_runtime_dbg_name "${${runtime_dbg_var}}" NAME)

    # copy release
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${${runtime_var}}"
                "${CMAKE_BINARY_DIR}/${dest_dir}/${_runtime_name}"
        RESULT_VARIABLE _copy_runtime_result
    )

    # copy debug
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${${runtime_dbg_var}}"
                "${CMAKE_BINARY_DIR}/${dest_dir}/${_runtime_dbg_name}"
        RESULT_VARIABLE _copy_runtime_dbg_result
    )

    # update cache if ok
    if(_copy_runtime_result EQUAL 0)
        set(${runtime_var} "${CMAKE_BINARY_DIR}/${dest_dir}/${_runtime_name}" CACHE FILEPATH "FBXSDK runtime" FORCE)
    else()
        message(FATAL_ERROR "Failed to copy ${runtime_var} from ${${runtime_var}}")
    endif()

    if(_copy_runtime_dbg_result EQUAL 0)
        set(${runtime_dbg_var} "${CMAKE_BINARY_DIR}/${dest_dir}/${_runtime_dbg_name}" CACHE FILEPATH "FBXSDK debug runtime" FORCE)
    else()
        message(FATAL_ERROR "Failed to copy ${runtime_dbg_var} from ${${runtime_dbg_var}}")
    endif()
endfunction()


if (DMT_OS_WINDOWS)
    set(_fbxsdk_vstudio_version "vs2022")
    set(_fbxsdk_approot "C:/Program Files/Autodesk/FBX/FBX SDK/2020.3.7")
    set(_fbxsdk_libdir_debug "lib/x64/debug")
    set(_fbxsdk_libdir_release "lib/x64/release")

    # libs + dll names
    set(_fbxsdk_libname_debug "libfbxsdk-md.lib")
    set(_fbxsdk_libname_release "libfbxsdk-md.lib")
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

    find_library(FBXSDK_LIBRARY ${_fbxsdk_libname_release}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_LIBRARY_DEBUG ${_fbxsdk_libname_debug}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

    find_library(FBXSDK_RUNTIME ${_fbxsdk_dll_name}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_release})
    find_library(FBXSDK_RUNTIME_DEBUG ${_fbxsdk_dll_name}
                 PATHS ${_fbxsdk_root}
                 PATH_SUFFIXES ${_fbxsdk_libdir_debug})

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
        fbxsdk_copy_runtime(FBXSDK_RUNTIME FBXSDK_RUNTIME_DEBUG "bin")

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
        fbxsdk_copy_runtime(FBXSDK_RUNTIME FBXSDK_RUNTIME_DEBUG "lib")

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
