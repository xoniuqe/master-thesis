cmake_minimum_required (VERSION 3.15)

#add_compile_definitions(HAVE_INLINE)
#add_compile_definitions(GSL_DISABLE_DEPRECATED)

add_subdirectory(extern/pybind)

#find_package(PythonLibs 3)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
#find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
#include_directories(../../include)


#SET(BUILD_SHARED_LIBS ON)
pybind11_add_module(stedepy main.cpp)


#add_compile_definitions(stedepy PUBLIC USE_ARMADILLO) 
target_link_libraries(stedepy  PRIVATE Eigen3::Eigen SteepestDescentLib ${ARMADILLO_LIBRARIES})

