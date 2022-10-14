# CMakeList.txt: CMake-Projektdatei der obersten Ebene. Führen Sie hier die globale Konfiguration aus,
# und schließen Sie Unterprojekte ein.
#
cmake_minimum_required (VERSION 3.14)

project(
  SteepestDescent
  VERSION 1.0
  LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  MESSAGE("Clang")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE("GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  MESSAGE("Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  MESSAGE("MSVC")
endif()

option(STEDEPY_TEST "Enable testing" ON)
option(STEDEPY_BENCHMARK "Enable benchmarks" ON)
option(STEDEPY_BUILD_MATLAB "Builds the mex functions" OFF)
option(STEDEPY_BUILD_PYTHON_MODULE "Builds the python module" ON)
option(STEDEPY_EXTERN_TBB "TBB located somewhere else" OFF)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Testing only available if this is the main app
  # Note this needs to be done in the main CMakeLists
  # since it calls enable_testing, which must be in the
  # main CMakeLists.
  include(CTest)
endif()

if(STEDEPY_EXTERN_TBB)
else()
add_subdirectory(extern/oneTBB)
endif()
unset(STEDEPY_EXTERN_TBB CACHE)
add_subdirectory(src)
add_subdirectory(api)
add_subdirectory (SteepestDescent)


# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR STEDEPY_TEST)
  AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR STEDEPY_BENCHMARK))
  add_subdirectory(benchmark)
endif()
