# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${ModernCMakeExample_SOURCE_DIR}/include/modern/*.hpp")
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS  "${PROJECT_SOURCE_DIR}/include/steepest_descent/*.h")
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/steepest_descent/datatypes.h" 
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/gauss_laguerre.h" 
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/math_utils.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/path_utils.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/integration/gsl_integrator.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/integration/gsl_integrator_2d.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/integral_1d.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/integral_2d.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/steepest_descent.h" 
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/configuration.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/api.h")

# Make an automatic library - will be static or dynamic based on user setting

#pybind stuff:
#add_library(objlib OBJECT gauss_laguerre.cpp math_utils.cpp path_utils.cpp integral.cpp steepest_descent.cpp  ${HEADER_LIST})


# This depends on (header only) boost
#target_link_libraries(objlib PRIVATE ${ARMADILLO_LIBRARIES})
# We need this directory, and users of our library will need it too
#target_include_directories(objlib PUBLIC ../include)

# shared libraries need PIC
#set_property(TARGET objlib PROPERTY POSITION_INDEPENDENT_CODE 1)

# shared and static libraries built from the same object files
#add_library(SteepestDescentLib SHARED $<TARGET_OBJECTS:objlib>)
#add_library(SteepestDescentLib_static STATIC $<TARGET_OBJECTS:objlib>)
#pybind stuff end

add_library(SteepestDescentLib gauss_laguerre.cpp math_utils.cpp path_utils.cpp ${HEADER_LIST}  integral_1d.cpp integral_2d.cpp steepest_descent.cpp)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")  # <--- this is needed to replace the default
find_package(GSL REQUIRED)
set(TBB_FIND_RELEASE_ONLY FALSE)
set(TBB_TEST FALSE)
set(TBB_USE_DEBUG FALSE)
find_package(TBB REQUIRED)

add_compile_definitions(HAVE_INLINE)
add_compile_definitions(GSL_DISABLE_DEPRECATED)

set_property(TARGET SteepestDescentLib PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

# We need this directory, and users of our library will need it too
target_include_directories(SteepestDescentLib PUBLIC ../include)

IF(WIN32)
set(BLA_STATIC true)
#target_include_directories(SteepestDescentLib PRIVATE ../extern/oneTbb/include)
#target_link_libraries(SteepestDescentLib Private ${ONETBB_LIBRARIES})

find_package(OpenBLAS)
find_package(LAPACK)
find_package(Armadillo REQUIRED)
target_link_libraries(SteepestDescentLib PRIVATE GSL::gsl LAPACK::LAPACK OpenBLAS::OpenBLAS ${ARMADILLO_LIBRARIES} TBB::tbb)
ELSE()
# This depends on (header only) boost
target_link_libraries(SteepestDescentLib PRIVATE GSL::gsl ${ARMADILLO_LIBRARIES} TBB::tbb)
ENDIF()
#if (MSVC)
#set_target_properties(SteepestDescentLib PROPERTIES LINK_FLAGS "/PROFILE")
#endif()
#target_compile_options(SteepestDescentLib PRIVATE -pedantic -Wall -Wextra -Wconversion)

IF(MSVC) 
	target_compile_options(SteepestDescentLib PRIVATE
	  /DARMA_DONT_USE_WRAPPER
	 # /static
	 # /pedantic
	  /W4
	  /BUILD_STATIC_LIBS
	  /GS
	  $<$<CONFIG:DEBUG>:/g3>
	  $<$<CONFIG:RELEASE>:/O2>
	)
ELSEIF(GCC)
	target_compile_options(SteepestDescentLib PRIVATE
	 # -DARMA_DONT_USE_WRAPPER
	  -BUILD_STATIC_LIBS
	  -pedantic
	  -Wall
	  -Wextra
	  -Wconversion
	  -Wsign-conversion
	  $<$<CONFIG:DEBUG>:-g3>
	  $<$<CONFIG:DEBUG>:-Og>
	  $<$<CONFIG:RELEASE>:-O3>
	)
ELSEIF()
ENDIF()

add_compile_definitions(
  $<$<CONFIG:DEBUG>:DEBUG>
)

#pybind stuff:
#target_include_directories(SteepestDescentLib PUBLIC ../include)
#target_include_directories(SteepestDescentLib_static PUBLIC ../include)

#target_compile_definitions(SteepestDescentLib PRIVATE "static")
# All users of this library will need at least C++11
#target_compile_features(SteepestDescentLib PUBLIC cxx_std_20)