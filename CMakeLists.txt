# CMakeList.txt: CMake-Projektdatei der obersten Ebene. Führen Sie hier die globale Konfiguration aus,
# und schließen Sie Unterprojekte ein.
#
cmake_minimum_required (VERSION 3.8)

project ("SteepestDescent")
set (CMAKE_CXX_STANDARD 20)
find_package(GSL REQUIRED)
# Schließen Sie Unterprojekte ein.
add_subdirectory ("SteepestDescent")
