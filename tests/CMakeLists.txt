# Testing library
Include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.9)

FetchContent_MakeAvailable(catch2)
# Adds Catch2::Catch2

find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})



# Tests need to be added as executables first
add_executable(tests tests_main.cpp math_lib_tests.cpp gauss_laguerre_tests.cpp)
target_compile_features(tests PRIVATE cxx_std_17)

# I'm using C++17 in the test
#target_compile_features(math_lib_tests PRIVATE cxx_std_17)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(tests PRIVATE SteepestDescentLib ${ARMADILLO_LIBRARIES} Catch2::Catch2)

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
add_test(NAME test_math_lib_tests COMMAND tests -r junit -o result.xml) # Command can be a target
