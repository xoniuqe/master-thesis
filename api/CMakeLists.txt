cmake_minimum_required (VERSION 3.14)

if(STEDEPY_BUILD_PYTHON_MODULE)
	add_subdirectory(python)
endif()

if(STEDEPY_BUILD_MATLAB)
	add_subdirectory(matlab)
endif()