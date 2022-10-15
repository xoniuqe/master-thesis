#project(StedepyMatlabPlugin)
#	cmake_minimum_required (VERSION 3.15)

find_package(Matlab REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


matlab_add_mex(NAME stedepyMatlab_2d MODULE SRC integrate_2d.cpp R2018a  LINK_TO SteepestDescentLib ${ARMADILLO_LIBRARIES})

matlab_add_mex(NAME stedepyMatlab_1d MODULE SRC integrate_1d.cpp R2018a  LINK_TO SteepestDescentLib ${ARMADILLO_LIBRARIES})
