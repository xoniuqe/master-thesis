# CMakeList.txt: CMake-Projekt für "SteepestDescent". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.14)
#add_compile_definitions(HAVE_INLINE)
#add_compile_definitions(GSL_DISABLE_DEPRECATED)

find_package(Armadillo REQUIRED)
#find_package(MKL REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#include_directories(${MKL_INCLUDE_DIRS})
#add_compile_definitions(ARMADILLO ARMA_DONT_USE_WRAPPER)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (SteepestDescent "SteepestDescent.cpp" "SteepestDescent.h")



#target_compile_features(SteepestDescent PRIVATE cxx_std_17)


add_compile_definitions(SteepestDescent PUBLIC USE_ARMADILLO)
# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.

target_link_libraries(SteepestDescent PRIVATE SteepestDescentLib ${ARMADILLO_LIBRARIES})

# from merge
#target_link_libraries(SteepestDescent PRIVATE GSL::gsl ${ARMADILLO_LIBRARIES})# Boost::Multiprecision)
#target_link_libraries(foo ${ARMADILLO_LIBRARIES})
#target_link_libraries(SteepestDescent PRIVATE GSL::gsl GSL::gslcblas)
