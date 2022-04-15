# CMakeList.txt: CMake-Projekt für "SteepestDescent". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)
add_compile_definitions(HAVE_INLINE)
add_compile_definitions(GSL_DISABLE_DEPRECATED)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (SteepestDescent "SteepestDescent.cpp" "SteepestDescent.h" "math_utils.h")

# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
target_link_libraries(SteepestDescent PRIVATE GSL::gsl ${ARMADILLO_LIBRARIES})# Boost::Multiprecision)
#target_link_libraries(foo ${ARMADILLO_LIBRARIES})
#target_link_libraries(SteepestDescent PRIVATE GSL::gsl GSL::gslcblas)