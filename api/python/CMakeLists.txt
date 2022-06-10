cmake_minimum_required (VERSION 3.15)

add_compile_definitions(HAVE_INLINE)
add_compile_definitions(GSL_DISABLE_DEPRECATED)


find_package(Armadillo REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})



pybind11_add_module(python_api main.cpp)


add_compile_definitions(python_api PUBLIC USE_ARMADILLO) 
target_link_libraries(python_api  PRIVATE Eigen3::Eigen GSL::gsl SteepestDescentLib ${ARMADILLO_LIBRARIES} )
