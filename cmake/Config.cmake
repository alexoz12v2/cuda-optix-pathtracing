################################################################################
# Config.cmake
#  - platform variable defition
#  - compiler variables definition
#  - functions to configure compiler flags on a per target basis
################################################################################
include_guard()

macro(dmt_define_environment)
  # -- detect the CPU architecture, and if different from x86_64, bail out --
  # while there are some CMake variables like CMAKE_SYSTEM_PROCESSOR and CMAKE_HOST_SYSTEM_PROCESSOR,
  # they are inconsistent (see docs). Therefore, use a small C file to detect architecture and report 
  # it in the compilation error message, which is then extracted and processed
  set(archdetect_c_code "
  #if defined(__aarch64__) || defined(__arm64) || defined(__arm64__) || defined(_M_ARM64)
      #error cmake_ARCH aarch64
  #elif defined(__arm__) || defined(__TARGET_ARCH_ARM)
      #if defined(__ARM_ARCH_9__) || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM-0 >= 9)
          #error cmake_ARCH armv9
      #elif defined(__ARM_ARCH_8__) || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM-0 >= 8)
          #error cmake_ARCH armv8
      #elif defined(__ARM_ARCH_7__) \\
          || defined(__ARM_ARCH_7A__) \\
          || defined(__ARM_ARCH_7R__) \\
          || defined(__ARM_ARCH_7M__) \\
          || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM-0 >= 7)
          #error cmake_ARCH armv7
      #elif defined(__ARM_ARCH_6__) \\
          || defined(__ARM_ARCH_6J__) \\
          || defined(__ARM_ARCH_6T2__) \\
          || defined(__ARM_ARCH_6Z__) \\
          || defined(__ARM_ARCH_6K__) \\
          || defined(__ARM_ARCH_6ZK__) \\
          || defined(__ARM_ARCH_6M__) \\
          || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM-0 >= 6)
          #error cmake_ARCH armv6
      #elif defined(__ARM_ARCH_5TEJ__) \\
          || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM-0 >= 5)
          #error cmake_ARCH armv5
      #else
          #error cmake_ARCH arm
      #endif
  #elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
      #error cmake_ARCH i386
  #elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
    #error cmake_ARCH x86_64 
  #elif defined(__ia64) || defined(__ia64__) || defined(_M_IA64)
      #error cmake_ARCH ia64
  #elif defined(__riscv) || defined(__riscv__) || defined(__riscv64__)
      #if defined(__riscv64__)
          #error cmake_ARCH riscv64
      #else
          #error cmake_ARCH riscv
      #endif
  #elif defined(__mips__) || defined(__mips) || defined(_M_MRX000)
      #if defined(__mips64) || defined(__mips64__)
          #error cmake_ARCH mips64
      #else
          #error cmake_ARCH mips
      #endif
  #elif defined(__ppc__) || defined(__ppc) || defined(__powerpc__) \\
        || defined(_ARCH_COM) || defined(_ARCH_PWR) || defined(_ARCH_PPC)  \\
        || defined(_M_MPPC) || defined(_M_PPC)
      #if defined(__ppc64__) || defined(__powerpc64__) || defined(__64BIT__)
          #error cmake_ARCH ppc64
      #else
          #error cmake_ARCH ppc
      #endif
  #elif defined(__s390__) || defined(__s390x__)
      #if defined(__s390x__)
          #error cmake_ARCH s390x
      #else
          #error cmake_ARCH s390
      #endif
  #endif

  #error cmake_ARCH unknown
  ")

  if(APPLE AND CMAKE_OSX_ARCHITECTURES)
   # On OS X we use CMAKE_OSX_ARCHITECTURES *if* it was set
    # First let's normalize the order of the values

    # Note that it's not possible to compile PowerPC applications if you are using
    # the OS X SDK version 10.6 or later - you'll need 10.4/10.5 for that, so we
    # disable it by default
    # See this page for more information:
    # http://stackoverflow.com/questions/5333490/how-can-we-restore-ppc-ppc64-as-well-as-full-10-4-10-5-sdk-support-to-xcode-4

    # Architecture defaults to i386 or ppc on OS X 10.5 and earlier, depending on the CPU type detected at runtime.
    # On OS X 10.6+ the default is x86_64 if the CPU supports it, i386 otherwise.

    foreach(osx_arch ${CMAKE_OSX_ARCHITECTURES})
      if("${osx_arch}" STREQUAL "ppc" AND ppc_support)
        set(osx_arch_ppc TRUE)
      elseif("${osx_arch}" STREQUAL "i386")
        set(osx_arch_i386 TRUE)
      elseif("${osx_arch}" STREQUAL "x86_64")
        set(osx_arch_x86_64 TRUE)
      elseif("${osx_arch}" STREQUAL "ppc64" AND ppc_support)
        set(osx_arch_ppc64 TRUE)
      else()
        message(FATAL_ERROR "Invalid OS X arch name: ${osx_arch}")
      endif()
    endforeach()

    # Now add all the architectures in our normalized order
    if(osx_arch_ppc)
      list(APPEND ARCH ppc)
    endif()

    if(osx_arch_i386)
      list(APPEND ARCH i386)
    endif()

    if(osx_arch_x86_64)
      list(APPEND ARCH x86_64)
    endif()

    if(osx_arch_ppc64)
      list(APPEND ARCH ppc64)
    endif()
  else()
    file(WRITE "${CMAKE_BINARY_DIR}/arch.c" "${archdetect_c_code}")

    try_run(
      run_result_unused
      compile_result_unused
      "${CMAKE_BINARY_DIR}"
      "${CMAKE_BINARY_DIR}/arch.c"
      COMPILE_OUTPUT_VARIABLE ARCH
      CMAKE_FLAGS CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
    )

    # Parse the architecture name from the compiler output
    string(REGEX MATCH "cmake_ARCH ([a-zA-Z0-9_]+)" ARCH "${ARCH}")

    # Get rid of the value marker leaving just the architecture name
    string(REPLACE "cmake_ARCH " "" ARCH "${ARCH}")

    # If we are compiling with an unknown architecture this variable should
    # already be set to "unknown" but in the case that it's empty (i.e. due
    # to a typo in the code), then set it to unknown
    if(NOT ARCH)
      set(ARCH unknown)
    endif()
  endif()

  string(TOUPPER "{ARCH}" ARCH_UPPER)
  set("DMT_ARCH_${ARCH_UPPER}" 1)

  if(NOT ${DMT_ARCH_X86_64})
    message(FATAL_ERROR "Only architecture supported is x86_64")
    return()
  endif()
  set(DMT_ARCH "DMT_ARCH_X86_64")

  # -- OS Detection (Cmake Variable CMAKE_SYSTEM_NAME) --
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(DMT_OS_WINDOWS 1)
    set(DMT_OS "DMT_OS_WINDOWS")
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(DMT_OS_LINUX 1)
    set(DMT_OS "DMT_OS_LINUX")
  else()
    message(FATAL_ERROR "We only support Linux and Windows based operating systems")
  endif()

  # -- compiler detection
  message(STATUS "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
  if(MSVC)
    set(DMT_COMPILER_MSVC 1)
    set(DMT_COMPILER "DMT_COMPILER_MSVC")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(DMT_COMPILER_CLANG_CL 1)
      message(STATUS "Compiler Found: clang-cl.exe")
    else()
      message(STATUS "Compiler Found: cl.exe")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DMT_COMPILER_CLANG 1)
    set(DMT_COMPILER "DMT_COMPILER_CLANG")
    message(STATUS "Compiler Found: clang++")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(DMT_COMPILER_GCC 1)
    set(DMT_COMPILER "DMT_COMPILER_GCC")
    message(STATUS "Compiler Found: g++")
  else()
    set(DMT_COMPILER "DMT_COMPILER_UNKNOWN")
    message(WARNING "Unrecognized compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  # -- build type definition
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DMT_BUILD_TYPE "DMT_DEBUG")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(DMT_BUILD_TYPE "DMT_RELEASE")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(DMT_BUILD_TYPE "DMT_DEBUG")
  elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(DMT_BUILD_TYPE "DMT_RELEASE")
  endif()
endmacro()


# Unused for now
function(dmt_get_msvc_flags out_flags)
  set(out_flags "")

  # Warning flags from dmt_set_target_warnings
  if(DMT_WARNINGS_AS_ERRORS) 
    list(APPEND out_flags "/WX")
  endif()

  list(APPEND out_flags
    "/w14242" "/w14254" "/w14263" "/w14265" "/w14287"
    "/we4289" "/w14296" "/w14311" "/w14545" "/w14546"
    "/w14547" "/w14549" "/w14555" "/w14619" "/w14640"
    "/w14826" "/w14905" "/w14906" "/w14928" "/permissive-"
    "/wd4068" "/wd4505" "/wd4800" "/wd4275"
  )

  # Optimization flags from dmt_set_target_optimization
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND out_flags "/Od" "/Zi")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND out_flags "/O2")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    list(APPEND out_flags "/O2" "/Zi")
  elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    list(APPEND out_flags "/O1")
  endif() 

  return(PROPAGATE out_flags)
endfunction()


# if you need to install libraries, it's best to provide in the repo a virtual environment
function(dmt_find_python_executable)
  if(DMT_OS_WINDOWS)
    execute_process(
      COMMAND py -3.11 -c "import sys; print(sys.executable)"
      RESULT_VARIABLE PYTHON_RESULT
      OUTPUT_VARIABLE PYTHON_EXEC
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NOT PYTHON_RESULT EQUAL 0)
      message(FATAL_ERROR "Python 3.11+ not found. Please install it with \"winget install python.python.3.11\" and ensure it's accessible via 'py -3.11'.")
    endif()
  else()
    execute_process(
      COMMAND python3 -c "import sys; print(sys.executable)"
      RESULT_VARIABLE PYTHON_RESULT
      OUTPUT_VARIABLE PYTHON_EXEC
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NOT PYTHON_RESULT EQUAL 0)
      message(FATAL_ERROR "Python 3.11+ not found. Please install it and ensure 'python3' is accessible.")
    endif()
    
    execute_process(
      COMMAND python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"
      RESULT_VARIABLE PYTHON_VERSION_OK
      ERROR_QUIET
    )
    
    if(NOT PYTHON_VERSION_OK EQUAL 0)
      message(FATAL_ERROR "Python version is lower than 3.11. Please upgrade Python.")
    endif()
  endif()
  return(PROPAGATE PYTHON_EXEC)
endfunction()


function(dmt_set_target_warnings target properties_visibility)
  option(DMT_WARNINGS_AS_ERRORS "Treat Compiler Warnings as errors" OFF)
  if(DMT_COMPILER_MSVC)
    if(DMT_WARNINGS_AS_ERRORS)
      target_compile_options(${target} ${properties_visibility} $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
    target_compile_options(${target} ${properties_visibility}
      $<$<COMPILE_LANGUAGE:CXX>:
        /w14242 # 'identifier': conversion from 'type1' to 'type1', possible loss of data
        /w14254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data
        /w14263 # 'function': member function does not override any base class virtual member function
        /w14265 # 'classname': class has virtual functions, but destructor is not virtual instances of this class may not be destructed correctly
        /w14287 # 'operator': unsigned/negative constant mismatch
        /we4289 # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside the for-loop scope
        /w14296 # 'operator': expression is always 'boolean_value'
        /w14311 # 'variable': pointer truncation from 'type1' to 'type2'
        /w14545 # expression before comma evaluates to a function which is missing an argument list
        /w14546 # function call before comma missing argument list
        /w14547 # 'operator': operator before comma has no effect; expected operator with side-effect
        /w14549 # 'operator': operator before comma has no effect; did you intend 'operator'?
        /w14555 # expression has no effect; expected expression with side- effect
        /w14619 # pragma warning: there is no warning number 'number'
        /w14640 # Enable warning on thread un-safe static member initialization
        /w14826 # Conversion from 'type1' to 'type_2' is sign-extended. This may cause unexpected runtime behavior.
        /w14905 # wide string literal cast to 'LPSTR'
        /w14906 # string literal cast to 'LPWSTR'
        /w14928 # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
        /permissive- # standards conformance mode

        # Disables, remove when appropriate
        /wd4068 # disable warnings about unknown pragmas (e.g. #pragma GCC)
        /wd4505 # disable warnings about unused functions that might be platform-specific
        /wd4800 # disable warnings regarding implicit conversions to bool
        /wd4275 # disable warnings about exporting non DLL-interface classes
      >
    )
  endif()

  if(DMT_COMPILER_GCC OR DMT_COMPILER_CLANG)
    if(DMT_WARNINGS_AS_ERRORS)
      target_compile_options(${target} ${properties_visibility} $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
    target_compile_options(${target} ${properties_visibility}
      $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps catch hard to track down memory errors
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output (ie printf)
        -Wimplicit-fallthrough # warn when a missing break causes control flow to continue at the next case in a switch statement
        -Wsuggest-override # warn when 'override' could be used on a member function overriding a virtual function
        -Wnull-dereference # warn if a null dereference is detected
        -Wold-style-cast # warn for c-style casts
        -Wpedantic # warn if non-standard C++ is used
      >
    )
  endif()

  if(DMT_COMPILER_GCC)
    # Don't enable -Wduplicated-branches for GCC < 8.1 since it will lead to false positives
    # https://github.com/gcc-mirror/gcc/commit/6bebae75035889a4844eb4d32a695bebf412bcd7
    target_compile_options(${target} ${properties_visibility}
      $<$<COMPILE_LANGUAGE:CXX>:
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
        # -Wuseless-cast # warn if you perform a cast to the same type (disabled because it is not portable as some type aliases might vary between platforms)
        $<$<VERSION_GREATER_EQUAL:${CMAKE_CXX_COMPILER_VERSION},8.1>:-Wduplicated-branches> # warn if if / else branches have duplicated code
      >
    )
  endif()

  if(DMT_COMPILER_CLANG OR DMT_COMPILER_CLANG_CL)
    target_compile_options(${target} ${properties_visibility}
      $<$<COMPILE_LANGUAGE:CXX>: -Wno-unknown-warning-option> # do not warn on GCC-specific warning diagnostic pragmas 
    )
  endif()

  # TODO move to another function
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  # set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON) # default = on for shared, off for static
  set_property(TARGET ${target} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
  # target_link_options(${target} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dlink>)
  target_compile_options(${target} ${properties_visibility} $<$<COMPILE_LANGUAGE:CUDA>:
    -dlink
    --cudart shared
    --cudadevrt static
    --expt-relaxed-constexpr # constexpr functions inside device code
    --extended-lambda # full C++20 lambda syntax inside device code
    -dc
    -use_fast_math
    # --relocatable-device-code true set by cuda separable compilation
  >)
endfunction()


function(dmt_set_target_optimization target properties_visibility)
  if (NOT properties_visibility MATCHES "INTERFACE")
    # https://learn.microsoft.com/en-us/cpp/error-messages/tool-errors/linker-tools-warning-lnk4098?view=msvc-170
    if(DEFINED DMT_OS_WINDOWS AND DEFINED DMT_COMPILER_MSVC)
      if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        # Debug Multithreaded DLL (/MDd)
        message(STATUS "(${target}) debug compilation detected on Windows MSVC. Linking to /MDd")
        target_link_options(${target} ${properties_visibility} $<$<COMPILE_LANGUAGE:CXX>:/NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrt.lib>)
        target_link_libraries(${target} ${properties_visibility} msvcrtd.lib)  # Explicitly link to msvcrtd.lib
      else()
        # Release Multithreaded DLL (/MD)
        message(STATUS "(${target}) release compilation detected on Windows MSVC. Linking to /MD")
        target_link_options(${target} ${properties_visibility} $<$<COMPILE_LANGUAGE:CXX>:/NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib>)
        target_link_libraries(${target} ${properties_visibility} msvcrt.lib)   # Explicitly link to msvcrt.lib
      endif()
    endif()


    if(CMAKE_BUILD_TYPE STREQUAL "Debug") # you can also use the CONFIG generator expression
      # NRVO has an issue with visual studio debugger, you cannot inspect the returned struct, hence disabling it
      # https://developercommunity.visualstudio.com/t/When-a-custom-function-returns-a-custom-/10422600
      target_compile_options(${target} ${properties_visibility}
        $<$<COMPILE_LANGUAGE:CXX>:
          $<$<BOOL:${DMT_COMPILER_MSVC}>:/Od /Zi /Zc:nrvo->
          $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-O0 -g -fprofile-arcs -ftest-coverage>
        >
        $<$<COMPILE_LANGUAGE:CUDA>: -G -g -O0>
      )
      target_link_options(${target} PUBLIC $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:--coverage>)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
      target_compile_options(${target} ${properties_visibility}
        $<$<COMPILE_LANGUAGE:CXX>:
          $<$<BOOL:${DMT_COMPILER_MSVC}>:/O2>
          $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-O3>
        >
      )
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      target_compile_options(${target} ${properties_visibility}
        $<$<COMPILE_LANGUAGE:CXX>:
          $<$<BOOL:${DMT_COMPILER_MSVC}>:/O2 /Zi>
          $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-O2 -g>
        >
        $<$<COMPILE_LANGUAGE:CUDA>: -G -g>
      )
    elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
      target_compile_options(${target} ${properties_visibility}
        $<$<COMPILE_LANGUAGE:CXX>:
          $<$<BOOL:${DMT_COMPILER_MSVC}>:/O1>
          $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-Os>
        >
      )
    endif()
  endif()
endfunction()


function(dmt_debug_print_target_props target name)
  get_target_property(dirs ${target} CXX_MODULE_DIRS)
  get_target_property(sset ${target} CXX_MODULE_SET)
  if(NOT name STREQUAL "")
    get_target_property(sset_name ${target} "CXX_MODULE_SET_${name}")
    get_target_property(dirs_name ${target} "CXX_MODULE_DIRS_${name}")
  endif()
  get_target_property(ssets ${target} CXX_MODULE_SETS)
  get_target_property(sstd ${target} CXX_MODULE_STD)
  get_target_property(scan ${target} CXX_SCAN_FOR_MODULES)
  message(STATUS "[${target}] CXX_MODULE_DIRS: ${dirs}")
  message(STATUS "[${target}] CXX_MODULE_SET: ${sset}")
  if(NOT name STREQUAL "")
    message(STATUS "[${target}] CXX_MODULE_SET_${name}: ${sset_name}")
    message(STATUS "[${target}] CXX_MODULE_DIRS_${name}: ${dirs_name}")
  endif()
  message(STATUS "[${target}] CXX_MODULE_SETS: ${ssets}")
  message(STATUS "[${target}] CXX_MODULE_STD: ${sstd}")
  message(STATUS "[${target}] CXX_SCAN_FOR_MODULES: ${scan}")
endfunction()


function(dmt_debug_print_exec_target_props target)
  get_target_property(icmcd ${target} IMPORTED_CXX_MODULES_COMPILE_DEFINITIONS)
  get_target_property(icmcf ${target} IMPORTED_CXX_MODULES_COMPILE_FEATURES)
  get_target_property(icmco ${target} IMPORTED_CXX_MODULES_COMPILE_OPTIONS)
  get_target_property(icmid ${target} IMPORTED_CXX_MODULES_INCLUDE_DIRECTORIES)
  get_target_property(icmll ${target} IMPORTED_CXX_MODULES_LINK_LIBRARIES)
  message(STATUS "[${target}] IMPORTED_CXX_MODULES_COMPILE_DEFINITIONS: ${icmcd}")
  message(STATUS "[${target}] IMPORTED_CXX_MODULES_COMPILE_FEATURES: ${icmcf}")
  message(STATUS "[${target}] IMPORTED_CXX_MODULES_COMPILE_OPTIONS: ${icmco}")
  message(STATUS "[${target}] IMPORTED_CXX_MODULES_INCLUDE_DIRECTORIES: ${icmcd}")
  message(STATUS "[${target}] IMPORTED_CXX_MODULES_LINK_LIBRARIES: ${icmll}")
endfunction()


# helper function to tweak visibility of public symbols
# ensure public symbols are hidden by default (exported ones are explicitly marked)
function(dmt_set_public_symbols_hidden target)
  set_target_properties(${target} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
endfunction()


function(dmt_add_compile_definitions target properties_visibility)
  if (NOT properties_visibility MATCHES "INTERFACE")
    if(DMT_OS_WINDOWS)
      set(DMT_PROJ_PATH ${PROJECT_SOURCE_DIR})
      string(REGEX REPLACE "/" "\\\\\\\\" DMT_PROJ_PATH ${DMT_PROJ_PATH})
    else()
      set(DMT_PROJ_PATH ${PROJECT_SOURCE_DIR})
    endif()
    target_compile_definitions(${target} ${properties_visibility} ${DMT_OS} "DMT_PROJ_PATH=\"${DMT_PROJ_PATH}\"" ${DMT_BUILD_TYPE} ${DMT_ARCH} ${DMT_COMPILER})
    if(DMT_OS_WINDOWS)
      target_compile_definitions(${target} ${properties_visibility} WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
      if(CMAKE_GENERATOR MATCHES "Visual Studio")
        target_compile_definitions(${target} ${properties_visibility} DMT_VS_STUPIDITY)
      endif()
    endif()
  endif()

  # <cassert>
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(${target} ${properties_visibility} NDEBUG)
  elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    target_compile_definitions(${target} ${properties_visibility} NDEBUG)
  endif() 
endfunction()


function(dmt_set_target_compiler_versions name property_visibility)
  #set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX) # this overrides CUDA's device linking step
  get_target_property(LINK_LANGUAGE ${name} LINKER_LANGUAGE)
  if (LINK_LANGUAGE STREQUAL "C")
    message(FATAL_ERROR "Target ${name} somehow has C linking semantics. Fix it")
  endif()
  # modules require C++20 support
  target_compile_features(${name} ${property_visibility} cxx_std_20 cuda_std_20)

  if(MSVC)
    target_compile_options(${name} ${property_visibility} $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  endif()
endfunction()


# usage: dmt_add_module_library(target sources...) -> sources in ARGN
# create a c++20 module library, with no target_sources preset, just initialize the bare necessities
# to have a fully functioning module
function(dmt_add_module_library name module_name)
  cmake_parse_arguments(PARSE_ARGV 0 arg "SHARED;INTERFACE" "" "")
  message(STATUS "Received arg: ${arg_SHARED}")
  # parse arguments and extract the clean target path name
  if(NOT "${THIS_ARGS_UNPARSED_ARGUMENTS}" STREQUAL "")
    message(FATAL_ERROR "unexpected arguments while calling dmt_add_module_library: ${THIS_ARGS_UNPARSED_ARGUMENTS}")
  endif()

  string(REPLACE "dmt-" "" target_path ${name})
  string(REPLACE "dmt-" "dmt::" alias_name ${name})

  message(STATUS "[${name}] target path name: ${target_path}, alias name: ${alias_name}")

  if(arg_SHARED)
    add_library(${name} SHARED)
  elseif(arg_INTERFACE)
    add_library(${name} INTERFACE)
  else()
    add_library(${name})
  endif()

  if(arg_INTERFACE)
    set(properties_visibility "INTERFACE")
    set(properties_visibility_public "INTERFACE")
  else()
    set(properties_visibility "PRIVATE")
    set(properties_visibility_public "PUBLIC")
  endif()

  dmt_set_target_compiler_versions(${name} ${properties_visibility})
  dmt_set_target_warnings(${name} ${properties_visibility})
  dmt_set_target_optimization(${name} ${properties_visibility})
  dmt_add_compile_definitions(${name} ${properties_visibility})

  # set the exported name of the target (the one you use to target_link_libraries) to dmt::{name}
  # I expect all targets to start with dmt. Replace all - with _,
  # define export symbol (for dlls, linking) as uppercase
  if(NOT properties_visibility MATCHES "INTERFACE")
    string(REPLACE "-" "_" NAME_UPPER "${name}")
    string(TOUPPER "${NAME_UPPER}" NAME_UPPER)

    target_compile_definitions(${name} PRIVATE ${NAME_UPPER}_EXPORTS)
    if(arg_SHARED)
      target_compile_definitions(${name} PUBLIC ${NAME_UPPER}_SHARED)
      set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    else()
      set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    endif()

    set_target_properties(${name} PROPERTIES EXPORT_NAME dmt::${target_path})

    set_target_properties(${name} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/lib>)
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/lib>)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>)
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "Modules")

  # Possible todo: Handle Shared libraries (SFML)

  # possible todo: Override PDB name/directory as 2 configurations generate debug symbols

  # add project include as include directory
  if(NOT properties_visibility MATCHES "INTERFACE")
    target_include_directories(${name}
      INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
      PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/${target_path} ${PROJECT_SOURCE_DIR}/include/${target_path} ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/extern
    )
  else()
    target_include_directories(${name}
      INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
  endif()

  # cmake's built-in clang-tidy doesn't seem to work with CXX_MODULES
  set_target_properties(${name} PROPERTIES CXX_CLANG_TIDY "")

  # create alias
  add_library(${alias_name} ALIAS ${name})
  dmt_debug_print_target_props(${name} ${module_name})
endfunction()


# This should be called only by the windows operating system
function(dmt_win32_add_custom_application_manifest target)
  if (NOT DEFINED DMT_COMPILER_MSVC)
    message(FATAL_ERROR "Application Manifest generation is supported only for `link.exe`")
  endif()
  # construct the full path to the executable file
  get_target_property(exec_path ${target} RUNTIME_OUTPUT_DIRECTORY)
  get_target_property(exec_name ${target} OUTPUT_NAME)
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    get_target_property(debug_postfix ${target} DEBUG_POSTFIX)
    string(APPEND exec_name ${debug_postfix})
  endif()
  get_target_property(exec_suffix ${target} SUFFIX)
  string(APPEND exec_path "\\" ${exec_name} ${exec_suffix})

  # set target options and define custom command
  target_link_options(${target} PRIVATE $<HOST_LINK:/MANIFEST:NO>)
  add_custom_command(TARGET ${target}
    POST_BUILD
    # Define the path to the temporary script file
    COMMAND set script_file="${CMAKE_CURRENT_BINARY_DIR}/run_manifest.ps1"

    # Write the PowerShell script to the file
    COMMAND ${CMAKE_COMMAND} -E echo "Write-Host 'Dot Sourcing embedding scripts'" > ${CMAKE_CURRENT_BINARY_DIR}/run_manifest.ps1
    COMMAND ${CMAKE_COMMAND} -E echo ". ${PROJECT_SOURCE_DIR}/scripts/embed_manifest.ps1" >> ${CMAKE_CURRENT_BINARY_DIR}/run_manifest.ps1
    COMMAND ${CMAKE_COMMAND} -E echo "Write-Host 'Invoking function'" >> ${CMAKE_CURRENT_BINARY_DIR}/run_manifest.ps1
    COMMAND ${CMAKE_COMMAND} -E echo "Invoke-Manifest-And-Embed -ExecutableFilePath '${exec_path}' -ManifestTemplateFilePath '${PROJECT_SOURCE_DIR}/res/win32-application.manifest' -ManifestTemplateParams @{version='1.0.0.0'; name='${exec_name}'; description='TEST DESCRIPTION'}" >> ${CMAKE_CURRENT_BINARY_DIR}/run_manifest.ps1

    # Run the generated PowerShell script
    COMMAND pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_BINARY_DIR}/run_manifest.ps1"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Embed custom application manifest ${target}"
    USES_TERMINAL
  )
endfunction()


function(dmt_add_cli target)
  add_executable(${target})
  set_target_properties(${target} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>
    DEBUG_POSTFIX -d
    OUTPUT_NAME "${target}"
    FOLDER "Examples/${v_target_name}"
    VS_DEBUGGER_WORKING_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>)
  # put the executable file in the right place
  # set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_directories(${target} PRIVATE $<1:${PROJECT_BINARY_DIR}/lib>)
  target_include_directories(${target} PRIVATE $<1:${PROJECT_BINARY_DIR}/lib>)

  dmt_set_target_compiler_versions(${target} PRIVATE)
  dmt_set_target_warnings(${target} PRIVATE)
  dmt_set_target_optimization(${target} PRIVATE)
  dmt_add_compile_definitions(${target} PRIVATE)
endfunction()


# usage: dmt_add_example creates an executable with no sources, configured and prepared to be
function(dmt_add_example target)
  string(REGEX REPLACE "^dmt-" "" v_target_name "${target}")
  string(REGEX REPLACE "-" "_" v_target_name "${v_target_name}")

  if(DEFINED DMT_OS_WINDOWS)
    add_executable(${target} WIN32)
    set_target_properties(${target} PROPERTIES 
      SUFFIX .exe
    )
  else()
    add_executable(${target})
  endif()

  set_target_properties(${target} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>
    DEBUG_POSTFIX -d
    OUTPUT_NAME "${target}"
    FOLDER "Examples/${v_target_name}"
    VS_DEBUGGER_WORKING_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>)

  # Create the target (assuming it's an executable)
  if(DEFINED DMT_OS_WINDOWS)
    # use /SUBSYSTEM:WINDOWS (which doesn't allocate a console when launched by double click and detaches itself from the console 
    # when launched from a `conhost` process)
    # add the PE executable with .com extension, since command line prefers it when calling a program without suffix extension
    # this will use /SUBSYSTEM:CONSOLE
    add_executable(${target}-launcher)
    target_sources(${target}-launcher PRIVATE ${PROJECT_SOURCE_DIR}/src/win32-launcher/launcher.cpp)
    # set the same properties for the launcher as well
    set_target_properties(${target}-launcher PROPERTIES 
      RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>
      DEBUG_POSTFIX -d
      FOLDER "Examples/${v_target_name}"
      VS_DEBUGGER_WORKING_DIRECTORY $<1:${PROJECT_BINARY_DIR}/bin>
      OUTPUT_NAME "${target}"
      PDB_NAME "${target}-launcher"
      SUFFIX .com
    )
    dmt_set_target_compiler_versions(${target}-launcher PRIVATE)
    dmt_set_target_warnings(${target}-launcher PRIVATE)
    dmt_set_target_optimization(${target}-launcher PRIVATE)
    dmt_add_compile_definitions(${target}-launcher PRIVATE)
    # force build system to rebuild the actual target when launcher is built
    add_dependencies(${target}-launcher ${target})

    dmt_win32_add_custom_application_manifest(${target})
    dmt_win32_add_custom_application_manifest(${target}-launcher)
  endif()

  message(STATUS "${target} ARGS_PUBLIC_SOURCES ${ARGS_PUBLIC_SOURCES}")
  message(STATUS "${target} ARGS_PRIVATE_SOURCES ${ARGS_PRIVATE_SOURCES}")
  message(STATUS "${target} ARGS_PUBLIC_DEPS ${ARGS_PUBLIC_DEPS}")
  message(STATUS "${target} ARGS_PRIVATE_DEPS ${ARGS_PRIVATE_DEPS}")

  # put the executable file in the right place
  # set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_directories(${target} PRIVATE $<1:${PROJECT_BINARY_DIR}/lib>)
  target_include_directories(${target} PRIVATE $<1:${PROJECT_BINARY_DIR}/lib>)

  dmt_set_target_compiler_versions(${target} PRIVATE)
  dmt_set_target_warnings(${target} PRIVATE)
  dmt_set_target_optimization(${target} PRIVATE)
  dmt_add_compile_definitions(${target} PRIVATE)

  # possible todo: Automatically include dependencies here
endfunction()


# usage: dmt_add_test target -> creates and adds a target which contains a test
# doesn't add any source
function(dmt_add_test target)
  cmake_parse_arguments(THIS_ARGS
    "" # no options
    "TARGET_LIST" # single argument keys
    "FILES;DEPENDENCIES" # multiple arguments keys
    ${ARGN}
  )
  message(STATUS "[dmt_add_test(${target})] TARGET_LIST ${THIS_ARGS_TARGET_LIST}")
  message(STATUS "[dmt_add_test(${target})] FILES ${THIS_ARGS_FILES}")
  message(STATUS "[dmt_add_test(${target})] DEPENDENCIES ${THIS_ARGS_DEPENDENCIES}")
  if(NOT DEFINED THIS_ARGS_FILES)
    message(FATAL_ERROR "[dmt_add_test] test target ${target} had no files: ${THIS_ARGS_FILES}")
  endif()
  if(NOT DEFINED ${THIS_ARGS_TARGET_LIST})
    message(FATAL_ERROR "[dmt_add_test] target list wasn't passed ${THIS_ARGS_TARGET_LIST}")
  endif()

  add_executable(${target})
  target_sources(${target} PRIVATE ${THIS_ARGS_FILES})
  # possible TODO: PCH

  target_link_directories(${target} PRIVATE $<1:${PROJECT_BINARY_DIR}/lib>)
  target_include_directories(${target} PRIVATE $<1:${PROJECT_BINARY_DIR}/lib>)
  set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # showup folder on visual studio
  set_target_properties(${target} PROPERTIES FOLDER "Tests")
  # startup path for debugging in IDEs
  set_target_properties(${target} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  dmt_set_target_compiler_versions(${target} PRIVATE)
  dmt_set_target_warnings(${target} PRIVATE)
  dmt_set_target_optimization(${target} PRIVATE)
  dmt_add_compile_definitions(${target} PRIVATE)

  dmt_set_public_symbols_hidden(${target})

  # dependencies
  if(DEFINED THIS_ARGS_DEPENDENCIES)
    target_link_libraries(${target} PRIVATE ${THIS_ARGS_DEPENDENCIES})
  endif()
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/extern ${PROJECT_SOURCE_DIR}/test/shared)

  # possible todo: dependencies and proper code coverage

  catch_discover_tests(${target})
  list(APPEND ${THIS_ARGS_TARGET_LIST} ${target})
  return(PROPAGATE ${THIS_ARGS_TARGET_LIST})
endfunction()


macro(dmt_add_subdirectories)
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # Get all subdirectories in the current source directory
  file(GLOB SUBDIRECTORIES RELATIVE ${BASE_DIR} ${BASE_DIR}/*)

  # Iterate over each subdirectory
  foreach(SUBDIR ${SUBDIRECTORIES})
    # Check if it is a directory
    if(IS_DIRECTORY ${BASE_DIR}/${SUBDIR})
      # Add the subdirectory
      message(STATUS "Adding subdirectory: ${SUBDIR}")
      add_subdirectory(${SUBDIR})
    endif()
  endforeach()
endmacro()
