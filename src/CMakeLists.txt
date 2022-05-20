# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${ModernCMakeExample_SOURCE_DIR}/include/modern/*.hpp")
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS  "${PROJECT_SOURCE_DIR}/include/steepest_descent/*.h")
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/steepest_descent/datatypes.h" 
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/gauss_laguerre.h" 
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/math_utils.h"
				"${PROJECT_SOURCE_DIR}/include/steepest_descent/path_utils.h")

# Make an automatic library - will be static or dynamic based on user setting


add_library(SteepestDescentLib gauss_laguerre.cpp math_utils.cpp path_utils.cpp ${HEADER_LIST} )

# We need this directory, and users of our library will need it too
target_include_directories(SteepestDescentLib PUBLIC ../include)

# This depends on (header only) boost
target_link_libraries(SteepestDescentLib PRIVATE ${ARMADILLO_LIBRARIES})

# All users of this library will need at least C++11
target_compile_features(SteepestDescentLib PUBLIC cxx_std_17)