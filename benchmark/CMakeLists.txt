# Testing library
Include(FetchContent)

set (BENCHMARK_ENABLE_INSTALL OFF)
set (BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(extern/benchmark)

# Adds google benchmark

find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})



# Tests need to be added as executables first
add_executable(benchmarks main.cpp benchmark_1d.cpp benchmark_2d.cpp)
target_compile_features(benchmarks PRIVATE cxx_std_17)

# I'm using C++17 in the test
#target_compile_features(math_lib_tests PRIVATE cxx_std_17)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(benchmarks PRIVATE SteepestDescentLib ${ARMADILLO_LIBRARIES} benchmark::benchmark)

