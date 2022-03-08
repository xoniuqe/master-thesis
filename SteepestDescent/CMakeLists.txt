# CMakeList.txt: CMake-Projekt für "SteepestDescent". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)
add_compile_definitions(HAVE_INLINE)
# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (SteepestDescent "SteepestDescent.cpp" "SteepestDescent.h")

# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
target_link_libraries(SteepestDescent PRIVATE GSL::gsl)
#target_link_libraries(SteepestDescent PRIVATE GSL::gsl GSL::gslcblas)