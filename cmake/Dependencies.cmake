include(FetchContent)

macro(dmt_setup_dependencies)
  if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.0 # or a later release
      GIT_SHALLOW    ON
    )
    FetchContent_MakeAvailable(Catch2)
    message(STATUS "Put catch2 on directory ${catch2_SOURCE_DIR}")
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif()

  if(NOT TARGET fmt::fmt)
    message(STATUS "FMT should be imported with module support ${FMT_MODULE}")
    FetchContent_Declare(fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG        10.1.1 # random version, should be fine
    )
    FetchContent_MakeAvailable(fmt)
  endif()

  if(NOT TARGET Backward::Backward)
    FetchContent_Declare(backward
      GIT_REPOSITORY https://github.com/bombela/backward-cpp
      GIT_TAG master  # or a version tag, such as v1.6
      SYSTEM          # optional, the Backward include directory will be treated as system directory
    )
    FetchContent_MakeAvailable(backward)
  endif()

  if(NOT TARGET glm::glm)
    FetchContent_Declare(
	    glm
	    GIT_REPOSITORY	https://github.com/g-truc/glm.git
	    GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
    )
    FetchContent_MakeAvailable(glm)
    add_compile_definitions(GLM_CONFIG_XYZW_ONLY=1)
  endif()
endmacro()
