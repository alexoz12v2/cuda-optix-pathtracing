#
# Helper function for finding the FBX SDK.
#
# sets: FBXSDK_FOUND, 
#       FBXSDK_DIR, 
#       FBXSDK_LIBRARY, 
#       FBXSDK_LIBRARY_DEBUG
#       FBXSDK_INCLUDE_DIR
#
set(_fbxsdk_version "2020.3.7")
set(_fbxsdk_vstudio_version "vs2022")

message("Looking for FBX SDK version: ${_fbxsdk_version}")

if (DMT_OS_WINDOWS)
    # the $ENV{PROGRAMFILES} variable doesn't really work since there's no 
    # 64-bit cmake version
    set(_fbxsdk_approot "C:/Program Files/Autodesk/FBX/FBX SDK")
    set(_fbxsdk_libdir_debug "lib/x64/debug")
    set(_fbxsdk_libdir_release "lib/x64/release")
    set(_fbxsdk_libname_debug "libfbxsdk-md.lib")
    set(_fbxsdk_libname_release "libfbxsdk-md.lib")

    # dependencies
    set(_fbxsdk_alembic_libname_debug "alembic-md.lib")
    set(_fbxsdk_alembic_libname_release "alembic-md.lib")
    set(_fbxsdk_libxml2_libname_debug "libxml2-md.lib")
    set(_fbxsdk_libxml2_libname_release "libxml2-md.lib")
    set(_fbxsdk_zlib_libname_debug "zlib-md.lib")
    set(_fbxsdk_zlib_libname_release "zlib-md.lib")
elseif (DMT_OS_LINUX)
    message(FATAL_ERROR "FIXME: find FBX SDK on Linux")
endif()

# should point the the FBX SDK installation dir
set(_fbxsdk_root "${_fbxsdk_approot}/${_fbxsdk_version}")
message("_fbxsdk_root: ${_fbxsdk_root}")

# find header dir and libs
find_path(FBXSDK_INCLUDE_DIR "fbxsdk.h" 
          PATHS ${_fbxsdk_root} 
          PATH_SUFFIXES "include")
message("FBXSDK_INCLUDE_DIR: ${FBXSDK_INCLUDE_DIR}")
find_library(FBXSDK_LIBRARY ${_fbxsdk_libname_release}
             PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_release})
message("FBXSDK_LIBRARY: ${FBXSDK_LIBRARY}")
find_library(FBXSDK_LIBRARY_DEBUG ${_fbxsdk_libname_debug}
             PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_debug})
message("FBXSDK_LIBRARY_DEBUG: ${FBXSDK_LIBRARY_DEBUG}")

# find dependencies
## alembic
find_library(FBXSDK_ALEMBIC_LIBRARY ${_fbxsdk_alembic_libname_debug}
             PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_release})
message("FBXSDK_ALEMBIC_LIBRARY: ${FBXSDK_ALEMBIC_LIBRARY}")
find_library(FBXSDK_ALEMBIC_LIBRARY_DEBUG ${_fbxsdk_alembic_libname_release}
             PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_debug})
message("FBXSDK_ALEMBIC_LIBRARY_DEBUG: ${FBXSDK_ALEMBIC_LIBRARY_DEBUG}")

## libxml2
find_library(FBXSDK_LIBXML2_LIBRARY ${_fbxsdk_libxml2_libname_release}
             PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_release})
message("FBXSDK_LIBXML2_LIBRARY: ${FBXSDK_LIBXML2_LIBRARY}")
find_library(FBXSDK_LIBXML2_LIBRARY_DEBUG ${_fbxsdk_libxml2_libname_debug}
	     PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_debug})
message("FBXSDK_LIBXML2_LIBRARY_DEBUG: ${FBXSDK_LIBXML2_LIBRARY_DEBUG}")

## zlib
find_library(FBXSDK_ZLIB_LIBRARY ${_fbxsdk_zlib_libname_release}
	     PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_release})
message("FBXSDK_ZLIB_LIBRARY: ${FBXSDK_ZLIB_LIBRARY}")
find_library(FBXSDK_ZLIB_LIBRARY_DEBUG ${_fbxsdk_zlib_libname_debug}
	     PATHS ${_fbxsdk_root}
             PATH_SUFFIXES ${_fbxsdk_libdir_debug})
message("FBXSDK_ZLIB_LIBRARY_DEBUG: ${FBXSDK_ZLIB_LIBRARY_DEBUG}")

if (FBXSDK_INCLUDE_DIR AND FBXSDK_LIBRARY AND FBXSDK_LIBRARY_DEBUG)
    set(FBXSDK_FOUND YES)
else()
    set(FBXSDK_FOUND NO)
endif()

# Create imported target if found
if (FBXSDK_FOUND)
    add_library(FBXSDK::fbxsdk STATIC IMPORTED)

    set_target_properties(FBXSDK::fbxsdk PROPERTIES
        IMPORTED_LOCATION_RELEASE "${FBXSDK_LIBRARY}"
        IMPORTED_LOCATION_DEBUG   "${FBXSDK_LIBRARY_DEBUG}"
        INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
    )

    add_library(FBXSDK::alembic STATIC IMPORTED)
    set_target_properties(FBXSDK::alembic PROPERTIES
        IMPORTED_LOCATION_RELEASE "${FBXSDK_ALEMBIC_LIBRARY}"
        IMPORTED_LOCATION_DEBUG   "${FBXSDK_ALEMBIC_LIBRARY_DEBUG}"
    )
    add_library(FBXSDK::libxml2 STATIC IMPORTED)
    set_target_properties(FBXSDK::libxml2 PROPERTIES
        IMPORTED_LOCATION_RELEASE "${FBXSDK_LIBXML2_LIBRARY}"
        IMPORTED_LOCATION_DEBUG   "${FBXSDK_LIBXML2_LIBRARY_DEBUG}"
    )
    add_library(FBXSDK::zlib STATIC IMPORTED)
    set_target_properties(FBXSDK::zlib PROPERTIES
        IMPORTED_LOCATION_RELEASE "${FBXSDK_ZLIB_LIBRARY}"
        IMPORTED_LOCATION_DEBUG   "${FBXSDK_ZLIB_LIBRARY_DEBUG}"
    )

    target_link_libraries(FBXSDK::fbxsdk INTERFACE FBXSDK::alembic FBXSDK::libxml2 FBXSDK::zlib)

    # Allow multi-config generators (e.g., Visual Studio) to pick debug/release automatically
    set_target_properties(FBXSDK::fbxsdk FBXSDK::alembic FBXSDK::libxml2 FBXSDK::zlib PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG   Debug
        MAP_IMPORTED_CONFIG_RELEASE Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
    )

    message(STATUS "FBX SDK imported target created: FBXSDK::fbxsdk")
endif()
