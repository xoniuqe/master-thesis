#project(StedepyMatlabPlugin)
cmake_minimum_required (VERSION 3.15)

find_package(Matlab)

matlab_add_mex(NAME stedepyMatlab MODULE SRC integrate_2d.cpp R2018a)