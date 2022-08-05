# benchmarking library
Include(FetchContent)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.0)

#set(BENCHMARK_ENABLE_GTEST_TESTS false)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES true)
#add_definitions( -UNDEBUG )
#add_definitions(-DTEST_BENCHMARK_LIBRARY_HAS_NO_ASSERTIONS)

if (NOT BUILD_SHARED_LIBS)
  add_definitions(-DBENCHMARK_STATIC_DEFINE)
endif()

FetchContent_MakeAvailable(benchmark)
find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})


add_executable(benchmarkTests integral2d_benchmark.cpp)

#target_compile_options(benchmarkTests PRIVATE
	 #-lbenchmark 
	 #-lpthread
#)

# I'm using C++17 in the test
#target_compile_features(math_lib_tests PRIVATE cxx_std_17)

target_link_libraries(benchmarkTests PUBLIC benchmark::benchmark_main PRIVATE SteepestDescentLib ${ARMADILLO_LIBRARIES})