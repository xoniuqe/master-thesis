# CMakeList.txt: CMake-Projekt für "SteepestDescent". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.14)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (SteepestDescent "SteepestDescent.cpp" "SteepestDescent.h")

#if (MSVC)
#set_target_properties(SteepestDescent PROPERTIES LINK_FLAGS "/PROFILE")
#endif()

#target_compile_features(SteepestDescent PRIVATE cxx_std_17)
#target_compile_options(SteepestDescent PRIVATE
	  #-pedantic
	  #-Wall
	  ##-Wextra
	  #-Wconversion
	  #-Wsign-conversion)

	#  set_property(TARGET SteepestDescent
     #        PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
IF(MSVC) 
	target_compile_options(SteepestDescent PRIVATE
	  #/DARMA_DONT_USE_WRAPPER
	 # /static
	 # /pedantic	  /W4
	  #/BUILD_STATIC_LIBS
	  #/GS
	  #/O2
	  #$<$<CONFIG:DEBUG>:/g3>
	  #$<$<CONFIG:RELEASE>:/O2>
	)
ELSEIF(GCC)
	target_compile_options(SteepestDescent PRIVATE
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

add_compile_definitions(SteepestDescent PUBLIC USE_ARMADILLO)
# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
 
target_link_libraries(SteepestDescent  PRIVATE SteepestDescentLib ${ARMADILLO_LIBRARIES} )
#pybind stuff:
#target_link_libraries(SteepestDescent  PRIVATE GSL::gsl SteepestDescentLib_static ${ARMADILLO_LIBRARIES} )


# from merge
#target_link_libraries(SteepestDescent PRIVATE GSL::gsl ${ARMADILLO_LIBRARIES})# Boost::Multiprecision)
#target_link_libraries(foo ${ARMADILLO_LIBRARIES})
#target_link_libraries(SteepestDescent PRIVATE GSL::gsl GSL::gslcblas)
