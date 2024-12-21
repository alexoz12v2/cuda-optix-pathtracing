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
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(DMT_COMPILER_CLANG_CL 1)
      message(STATUS "Compiler Found: clang-cl.exe")
    else()
      message(STATUS "Compiler Found: cl.exe")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DMT_COMPILER_CLANG 1)
    message(STATUS "Compiler Found: clang++")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(DMT_COMPILER_GCC 1)
    message(STATUS "Compiler Found: g++")
  else()
    message(WARNING "Unrecognized compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endmacro()


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


macro(dmt_set_target_warnings target)
  option(DMT_WARNINGS_AS_ERRORS "Treat Compiler Warnings as errors" OFF)
  if(DMT_COMPILER_MSVC)
    if(DMT_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
    target_compile_options(${target} PRIVATE
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
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Werror>)
    endif()
    target_compile_options(${target} PRIVATE
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
    target_compile_options(${target} PRIVATE
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
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>: -Wno-unknown-warning-option> # do not warn on GCC-specific warning diagnostic pragmas 
    )
  endif()
endmacro()


function(dmt_set_target_optimization target)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        $<$<BOOL:${DMT_COMPILER_MSVC}>:/Od /Zi>
        $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-O0 -g -fprofile-arcs -ftest-coverage>
      >
      $<$<COMPILE_LANGUAGE:CUDA>:
        -G
      >
    )
    target_link_options(${target} PUBLIC $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:--coverage>)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        $<$<BOOL:${DMT_COMPILER_MSVC}>:/O2>
        $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-O3>
      >
    )
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        $<$<BOOL:${DMT_COMPILER_MSVC}>:/O2 /Zi>
        $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-O2 -g>
      >
      $<$<COMPILE_LANGUAGE:CUDA>:
        -G
      >
    )
  elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        $<$<BOOL:${DMT_COMPILER_MSVC}>:/O1>
        $<$<OR:$<BOOL:${DMT_COMPILER_GCC}>,$<BOOL:${DMT_COMPILER_CLANG}>>:-Os>
      >
    )
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


function(dmt_add_compile_definitions target)
  if(DMT_OS_WINDOWS)
    set(DMT_PROJ_PATH ${PROJECT_SOURCE_DIR})
    string(REGEX REPLACE "/" "\\\\\\\\" DMT_PROJ_PATH ${DMT_PROJ_PATH})
  else()
    set(DMT_PROJ_PATH ${PROJECT_SOURCE_DIR})
  endif()
  target_compile_definitions(${target} PRIVATE ${DMT_OS} "DMT_PROJ_PATH=\"${DMT_PROJ_PATH}\"")
endfunction()


# usage: dmt_add_module_library(target sources...) -> sources in ARGN
# create a c++20 module library, with no target_sources preset, just initialize the bare necessities
# to have a fully functioning module
function(dmt_add_module_library name module_name)
  # parse arguments and extract the clean target path name
  cmake_parse_arguments(THIS_ARGS
    "" # no options
    "MODULE_INTERFACE;MODULE_IMPLEMENTATION" # single argument keys
    "MODULE_PARTITION_INTERFACES;MODULE_PARTITION_IMPLEMENTATIONS;HEADERS" # multiple arguments keys
    ${ARGN}
  )
  if(NOT "${THIS_ARGS_UNPARSED_ARGUMENTS}" STREQUAL "")
    message(FATAL_ERROR "unexpected arguments while calling dmt_add_module_library: ${THIS_ARGS_UNPARSED_ARGUMENTS}")
  endif()

  string(REPLACE "dmt-" "" target_path ${name})
  string(REPLACE "dmt-" "dmt::" alias_name ${name})
  if(NOT target_path STREQUAL ${module_name})
    message(WARNING "${target_path} not equal to ${module_name} (maybe target name is different from module name, and it shouldn't)")
  endif()

  message(STATUS "[${name}] MODULE_INTERFACE: ${THIS_ARGS_MODULE_INTERFACE}")
  message(STATUS "[${name}] MODULE_IMPLEMENTATION: ${THIS_ARGS_MODULE_IMPLEMENTATION}")
  message(STATUS "[${name}] MODULE_PARTITION_INTERFACES: ${THIS_ARGS_MODULE_PARTITION_INTERFACES}")
  message(STATUS "[${name}] MODULE_PARTITION_IMPLEMENTATIONS: ${THIS_ARGS_MODULE_PARTITION_IMPLEMENTATIONS}")
  message(STATUS "[${name}] HEADERS: ${THIS_ARGS_HEADERS}")
  message(STATUS "[${name}] target path name: ${target_path}, alias name: ${alias_name}")

  set(interface_file_list "${CMAKE_SOURCE_DIR}/include/${target_path}/${THIS_ARGS_MODULE_INTERFACE}")
  set(implementation_file_list "${THIS_ARGS_MODULE_IMPLEMENTATION}")
  set(header_file_list "")
  if(DEFINED THIS_ARGS_MODULE_PARTITION_INTERFACES)
    foreach(interface_file ${THIS_ARGS_MODULE_PARTITION_INTERFACES})
      string(PREPEND interface_file "${CMAKE_SOURCE_DIR}/include/${target_path}/")
      list(APPEND interface_file_list "${interface_file}")
    endforeach()
  endif()
  if (DEFINED THIS_ARGS_HEADERS)
    foreach(header_file ${THIS_ARGS_HEADERS})
      string(PREPEND header_file "${CMAKE_SOURCE_DIR}/include/${target_path}/")
      list(APPEND header_file_list "${header_file}")
    endforeach()
  endif()
  if(DEFINED THIS_ARGS_MODULE_PARTITION_IMPLEMENTATIONS)
    foreach(impl_file ${THIS_ARGS_MODULE_PARTITION_IMPLEMENTATIONS})
      list(APPEND implementation_file_list "${impl_file}")
    endforeach()
  endif()
  message(STATUS "${name} target sources are\n\tinterfaces     ${interface_file_list}\n\t"
    "implementation ${implementation_file_list}")


  # add library target (static)
  add_library(${name})

  # may give problems if CUDA interfaces between modules are shared
  set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
  # modules require C++20 support
  target_compile_features(${name} PUBLIC cxx_std_20)

  # cuda specific
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  dmt_set_target_warnings(${name})
  dmt_set_target_optimization(${name})
  dmt_add_compile_definitions(${name})

  # Possible TODO: Pre Compiled Headers

  if(MSVC)
    # TODO correct
    #set(BMI ${CMAKE_CURRENT_BINARY_DIR}/${name}.ifc)
    #target_compile_options(${name}
    #  PRIVATE /interface /ifcOutput ${BMI} # treat source as module interface unit https://www.modernescpp.com/index.php/c-20-module-interface-unit-and-module-implementation-unit/
    #   INTERFACE /reference ${module_name}=${BMI} # the module can reference an external interface file called ${module_name} with path ${BMI}
    # )
    # when the clean target is invoked, then remove all files in the ${BMI} path
    #set_target_properties(${name} PROPERTIES ADDITIONAL_CLEAN_FILES ${BMI})
    # mark the files as generated such that cmake doesn't expect them to be already there
    #set_source_files_properties(${BMI} PROPERTIES GENERATED ON)

    # setup path for .ifc file
    # set(BMI ${CMAKE_CURRENT_BINARY_DIR}/${module_name}.ifc)
    # message(STATUS "BMI we want to generate from target ${name} is ${BMI}")

    # set flag implementation unit
    #target_compile_options(${name} 
    #  PRIVATE /c /interface /TP /ifcOutput ${BMI}
    #  PUBLIC /reference ${module_name}=${BMI}
    #)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /Zc:preprocessor>)

    #get_target_property(thing ${name} CXX_SCAN_FOR_MODULES)
    #message("[${name}] CXX_SCAN_FOR_MODULES: ${thing}")

    # precompile module interface with a custom target (excluded from all?)
    # might not work with cuda interfaces
    # dmt_get_msvc_flags(out_flags)
    # get_target_property(target_flags ${name} COMPILE_FLAGS)
    # if(target_flags STREQUAL "target_flags-NOTFOUND")
    #  set(target_flags "")
    # endif()
    # message(STATUS ${out_flags})
    # message(STATUS ${target_flags})
    # add_custom_target("${name}_interface" 
    #  BYPRODUCTS ${BMI_PROD}
    #  COMMAND ${CMAKE_CXX_COMPILER} ${target_flags} ${out_flags} /std:c++20 /interface /ifcOutput ${BMI} /TP /c ${interface_file_list}
    #  DEPENDS ${interfaces}
    #)
    # add_dependencies(${name} "${name}_interface")
    # set_target_properties(${name} PROPERTIES ADDITIONAL_CLEAN_FILES ${BMI})
    # set_source_files_properties(${BMI} PROPERTIES GENERATED ON)
  elseif(DMT_COMPILER_GCC OR DMT_COMPILER_CLANG)
    #set(BMI ${CMAKE_CURRENT_BINARY_DIR}/${name}.pcm)
    #string(REPLACE "dmt-" "" dep_no_namespace ${BMI})
    #if(NOT dep_no_namespace STREQUAL "")
    #  set(BMI ${dep_no_namespace})
    #endif()
    #message("Compile Options for module target ${name} on BMI ${BMI}")
    # specify prebuilt module dependencies with -fprebuilt-module-path (TODO)

    # precompile prevents BMI generation, that's because I will generate it by myself?
    # target_compile_options(${name} PUBLIC -fmodules --precompile -fmodule-output=${BMI})

    #add_custom_command(
    #  OUTPUT ${BMI}
    #  COMMAND ${CMAKE_CXX_COMPILER} -std=c++20 -x -c++module --precompile -c -o ${BMI}
    #    ${CMAKE_SOURCE_DIR}/include/testdmt/testImplementationUnit.cppm
    #  DEPENDS ${CMAKE_SOURCE_DIR}/include/testdmt/testImplementationUnit.cppm
    #)
  endif()

  # set the exported name of the target (the one you use to target_link_libraries) to dmt::{name}
  # I expect all targets to start with dmt. Replace all - with _,
  # define export symbol (for dlls, linking) as uppercase
  string(REPLACE "-" "_" NAME_UPPER "${target_path}")
  string(TOUPPER "${NAME_UPPER}" NAME_UPPER)
  set_target_properties(${name} PROPERTIES DEFINE_SYMBOL ${NAME_UPPER}_EXPORTS)
  set_target_properties(${name} PROPERTIES EXPORT_NAME dmt::${target_path})

  # Possible todo: Handle Shared libraries (SFML)

  # possible todo: Override PDB name/directory as 2 configurations generate debug symbols

  # add project include as include directory
  target_include_directories(${name}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include/${target_path}
  )

  if(NOT DMT_CLANG_TIDY_COMMAND STREQUAL "")
    set_target_properties(${name} PROPERTIES CXX_CLANG_TIDY ${DMT_CLANG_TIDY_COMMAND})
  endif()
  if(DMT_COMPILER_CLANG AND NOT DMT_CLANG_TIDY_COMMAND STREQUAL "")
    # TODO: handle dependencies
    message(STATUS "clang-tidy requires me to manually setup the prebuilt-module-path,\n\tsetting path for ${name} to ${PROJECT_BINARY_DIR}/src/${target_path}/CMakeFiles/${name}.dir")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        -fprebuilt-module-path=${PROJECT_BINARY_DIR}/src/${target_path}/CMakeFiles/${name}.dir
      >
    )
  endif()

  # construct the sources lists
  list(LENGTH header_file_list header_count)
  if(header_count GREATER 0)
    target_sources(${name}
      PUBLIC
        FILE_SET ${module_name} TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/${target_path}
          FILES ${interface_file_list}
        FILE_SET "${module_name}_headers" TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/${target_path}
          FILES ${header_file_list}
      PRIVATE
        ${implementation_file_list}
    )
  else()
    target_sources(${name}
      PUBLIC
        FILE_SET ${module_name} TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/${target_path}
          FILES ${interface_file_list}
      PRIVATE
        ${implementation_file_list}
    )
  endif()

  # create alias
  add_library(${alias_name} ALIAS ${name})
  dmt_debug_print_target_props(${name} ${module_name})
  
  if(MSVC)
    if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
      message(FATAL_ERROR "Windows with Ninja doesn't find the module interface from the module implementation unit. I don't know why.")
      # module dependency between module interface unit and module implementation unit works fine on visual studio, 
      # but not on ninja. Still have no idea why
      #set(BMI ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name}.dir/${module_name}.ifc)
      #message("GENERATOR ------------------- ${CMAKE_GENERATOR}, different from VS. Trying to patch up module on ${BMI}")
      #target_compile_options(${name}
      #  PRIVATE /interface /ifcOutput ${BMI}
      #  PUBLIC /reference ${module_name}=${BMI}
      # )
    endif()
  endif()
endfunction()


# usage: dmt_add_example creates an executable with no sources, configured and prepared to be
function(dmt_add_example target)
  set(multivalue PUBLIC_SOURCES PRIVATE_SOURCES PUBLIC_DEPS PRIVATE_DEPS)
  cmake_parse_arguments(ARGS "" "" "${multivalue}" ${ARGN})

  # Create the target (assuming it's an executable)
  add_executable(${target})

  message(STATUS "${target} ARGS_PUBLIC_SOURCES ${ARGS_PUBLIC_SOURCES}")
  message(STATUS "${target} ARGS_PRIVATE_SOURCES ${ARGS_PRIVATE_SOURCES}")
  message(STATUS "${target} ARGS_PUBLIC_DEPS ${ARGS_PUBLIC_DEPS}")
  message(STATUS "${target} ARGS_PRIVATE_DEPS ${ARGS_PRIVATE_DEPS}")

  # Add public sources if provided
  if(ARGS_PUBLIC_SOURCES)
    target_sources(${target} PUBLIC ${ARGS_PUBLIC_SOURCES})
  endif()

  # Add private sources if provided
  if(ARGS_PRIVATE_SOURCES)
    target_sources(${target} PRIVATE ${ARGS_PRIVATE_SOURCES})
  endif()

  # Add public dependencies if provided
  if(ARGS_PUBLIC_DEPS)
    target_link_libraries(${target} PUBLIC ${ARGS_PUBLIC_DEPS})
  endif()

  # Add private dependencies if provided
  if(ARGS_PRIVATE_DEPS)
    target_link_libraries(${target} PRIVATE ${ARGS_PRIVATE_DEPS})
  endif()

  # possible todo: PCH

  set_target_properties(${target} PROPERTIES DEBUG_POSTFIX -d)
  # target folder (will show in visual studio)
  set_target_properties(${target} PROPERTIES FOLDER "Examples")
  # visual studio startup path for debugging
  set_target_properties(${target} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  dmt_set_target_warnings(${target})
  dmt_set_target_optimization(${target})
  dmt_add_compile_definitions(${target})

  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /Zc:preprocessor>)
    #target_compile_options(${target} PRIVATE )
  endif()

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

  # showup folder on visual studio
  set_target_properties(${target} PROPERTIES FOLDER "Tests")
  # startup path for debugging in IDEs
  set_target_properties(${target} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(${target} PUBLIC cxx_std_20)

  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /Zc:preprocessor>)
  endif()

  # Iterate over dependencies to add their corresponding BMI paths
  #set(BMI "")
  #set(IS_FIRST_DEPENDENCY TRUE)
  #set(FILTERED_DEPENDS ${THE_DEPENDS})
  #list(REMOVE_ITEM FILTERED_DEPENDS "dmt-test-main")
  #foreach(dep ${FILTERED_DEPENDS})
  #  # Remove the 'dmt::' namespace from the dependency if present
  #  string(REPLACE "dmt::" "" dep_no_namespace ${dep})
  #  if(NOT dep_no_namespace STREQUAL "")
  #    set(dep ${dep_no_namespace})
  #  endif()
  #  message(STATUS "current dep ${dep}")

  #  # Skip the first dependency handling the colon
  #  if(IS_FIRST_DEPENDENCY)
  #    set(BMI "${CMAKE_BINARY_DIR}/src/${dep}/${dep}.pcm")
  #    set(IS_FIRST_DEPENDENCY FALSE)
  #  else()
  #    # For subsequent dependencies, prepend the colon
  #    set(BMI "${BMI}:${CMAKE_BINARY_DIR}/src/${dep}/${dep}.pcm")
  #  endif()
  #endforeach()

  #message(STATUS "Target ${target} looks for dependent BMIs on paths ${BMI}")
  #if(DMT_COMPILER_GCC OR DMT_COMPILER_CLANG)
  #  target_compile_options(${target} PRIVATE
  #    -fmodules
  #  )
  #  if(NOT BMI STREQUAL "")
  #    target_compile_options(${target} PRIVATE
  #      -fprebuilt-module-path=${BMI}
  #    )
  #  endif()
  #endif()

  dmt_set_target_warnings(${target})
  dmt_set_target_optimization(${target})
  dmt_set_public_symbols_hidden(${target})
  dmt_add_compile_definitions(${target})

  # dependencies
  if(DEFINED THIS_ARGS_DEPENDENCIES)
    target_link_libraries(${target} PRIVATE ${THIS_ARGS_DEPENDENCIES})
  endif()
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/extern ${PROJECT_SOURCE_DIR}/test/shared)

  # possible todo: dependencies and proper code coverage

  catch_discover_tests(${target} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  message(STATUS "test: Discovered tests for target ${target} from directory ${CMAKE_CURRENT_LIST_DIR}")
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
