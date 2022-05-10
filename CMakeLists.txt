# CMakeList.txt: CMake-Projektdatei der obersten Ebene. Führen Sie hier die globale Konfiguration aus,
# und schließen Sie Unterprojekte ein.
#
cmake_minimum_required (VERSION 3.12)

project ("SteepestDescent")
set (CMAKE_CXX_STANDARD 20)


# Schließen Sie Unterprojekte ein.
add_subdirectory ("SteepestDescent")
