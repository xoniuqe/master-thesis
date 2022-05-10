# CMakeList.txt: CMake-Projekt für "SteepestDescent". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#

find_package(Armadillo REQUIRED)
#find_package(MKL REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})
#include_directories(${MKL_INCLUDE_DIRS})
#add_compile_definitions(ARMADILLO ARMA_DONT_USE_WRAPPER)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (SteepestDescent "SteepestDescent.cpp" "SteepestDescent.h" "math_utils.h" "gauss_laguerre.h" "datatypes.h" "gsl_vector_utils.h")
add_compile_definitions(SteepestDescent PUBLIC USE_ARMADILLO)
# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
target_link_libraries(SteepestDescent PRIVATE ${ARMADILLO_LIBRARIES})
