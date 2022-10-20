# Master thesis

## How to build

To run this project, first the repository has to be cloned.


### Prerequisits

* git
* cmake (>3.15)
* C++ compiler (MSVC, GCC, CLANG)

### Windows
Easiest setup on windows is to have Visual Studio 2022 installed with the options for CMake and Crossplattform projects.

Install  oneAPI oneTBB:  https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onetbb

CD to the root folder of the repo
Update the submodules:
git submodule sync --recursive
git submodule update --init --recursive

Bootstrap VCPKG
cd vcpkg
bootstrap-vcpkg.bat

Install GSL Eigen3 with VCPKG
vcpkg install gsl gsl:x64-windows  Eigen3:x64-windows
vcpkg integrate install

To compile either use VS2022 or run the CMAKE build

### Ubuntu
CD to the root folder of the repo

Update the submodules:
git submodule sync --recursive
git submodule update --init --recursive

Install the libraries to compile (Armadillo, OpenBLA, LApack, GSL, python3, eigen, tbb)
- apt-get update --yes
- apt-get install --yes cmake libarmadillo-dev libopenblas-dev liblapack-dev libgsl-dev python3-dev libeigen3-dev libtbb-dev


### Macos
    
### Build with CMAKE

build the project
Build wiht -DSTEDEPY_BUILD_MATLAB=TRUE to build the matlab plugin. This requires Matlab versions >= R2018a

- cmake -S . -B build -DTBB_TEST=FALSE
- cmake --build build -v 

This build the project into out/build/

## How to run

### From Visual Studio 2022

On Windows the project and some tests can be run from Visual Studio. If the WSL2 Subsystem with UBUNTU is installed the ubuntu setup can also be debugged.
In the fodler SteepestDescent are some test files.


### Matlab

After the project was build with cmake (and -DSTEDEPY_BUILD_MATLAB=TRUE ), the output folder should contain some compiled mexfiles (build/out/<Target>/api/matlab).
This file has to loaded in the matlab environment and then the functions can be run by the name of the files (StedepyMatlab_2d and StedepyMatlab_1d)

### Python
After the project was build with cmake , the output folder should contain the compiled  python module (build/out/<Target>/api/python).
Python3 can load this if it is in its search path, or in the current working directory.




## Setting up my project:
needs to be updated!
(gsl )
* clone the repo
* cd vcpkg
* bootstrap-vcpkg.bat
* vcpkg install gsl gsl:x64-windows 
 (add eigen3)
(todo: vcpkg install @ResponseFile)
* vcpkg integrate install

"D:/Master/SteepestDescent/vcpkg\vcpkg.exe" install TBB:x64-windows
"D:/Master/SteepestDescent/vcpkg\vcpkg.exe" install Eigen3:x64-windows
windows braucht wohl noch tbb wenn ich -DCMAKE_TOOLCHAIN_FILE=D:/Master/SteepestDescent/vcpkg/scripts/buildsystems/vcpkg.cmake nutze
DCMAKE_TOOLCHAIN_FILE=$(SolutionDir)/vcpkg/scripts/buildsystems/vcpkg.cmake

(armadillo)
* clone the repo
on windows:
 * install intel mkl
 *

 19.09.22

 install onetbb seems to fix building on windows!

 https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onetbb
#TODO

* impl 2d:
 - steepest descen in 2d and for Y axis, is slightly different than the 1d case
* refactor the integral parameters (1d => nodes and weights are calculatet in the operator() => this could be simply passed via the constructor)
* write benchmarks
* write python api
* write more tests


## Folder structure

Explain that !

## Thoughts

The nature of the problem allows it to be splitted in many sub calculations.
This should make it easier to run parts of this in threads or even in separate calls to the provided API with muliple machines (e.g. when scaling on a cluster)
This has still to be verified, but each integration runs over a small triangular area, so the splitting could be achieved by an external call.
We should provide each function in the planned APIs (Python / Matlab)


## MatLab-Implementation analysis

### (1) Calculating the laguerre nodes and weiths

This is done in the function lagpts. It takes the order n and calculates n weights and nodes which than get used to solve the integral.

### (2) Steepest Descent
Function GetSteepDec1

Uses the function f returned by WeightFunPerPath (3) to calculate the actual integral by using the gau�-laguerre quadrature rule. We construct paths along the spitting points Sp_1 and Sp_2.

With f we evaluate the complex function at the nodes calculated by (1) and weigh the results by the matching weights. The result is summed up and is the
wanted integral for the path.

### (3) WeightFunPerPath

Calls GetComplexPath for the given splitting point which returns the path and its derivative. With that we construct a function which evaluates the complex function at the given path.
In essence it calculates f(h(x)) for us, with x = [a,b].


## Implementation details

### Representation of the general inputs

Input Variable | Fields | Methods
---------------|--------|--------
to be named | A, b, r |  


method | input | result | used_methods
get_complex_roots | y, A, b, r | complex_root: {c_0, c}| calculate_P_x
calculate_P_x | x, y, A, b, r | double | -
get_complex_path | split_point, y(=), A(=), b(=), r(=), q, complex_root, sing_point(!) | path + derived path: {path_function, path_function} |calculate_P_x
get_weighted_path | split_point, y(=), A(=), b(=), r(=), q, k, s, complex_root(=), sing_point(=) | weighted path: path_function | get_complex_path, calculate_P_x
get_weighted_path_1d | split_point, q, k, s, function_Px, path, derivative | weighted path : path_function | (calculate_P_x via function pointer function_Px) 
integral_1d | A, b, r, mu, y, left_split, right_split | integral | gauss_laguerre::calculate_laguerre_points_and_weights, math_utils::get_complex_roots 
steepest_desc (constr) | nodes(=), weights(=), k(=), y(=), A(=), b(=), r(=), q(=), s(=), complex_root, sing_point | steepest_desc | -
steepest_desc.operator() | first_split, second_split | integral | path_utils::get_weightet_path, gauss_laguerre::calculate_integral_cauchy
Legend:
 * (=): Just passed through
 * (!): Just control flow


### Matlab plugin details

Implmented as so-called 'Mexfunctions'. Assumes RAII (explain that concept in a footnote?)
### Used technologies

Programming language: C++

Mathlibrary: Gnu Scientifc Library

Testing: Catch2

Benchmarking: to be decided

Speedup: Intel-Threading-Building-Blocks, OpenCl, Cuda(?)

### Speedup 

I will use OneTBB (Intel threading building blocks) and try to use OpenCl

### Requirements

1. The code should run faster than the Matlab implementation
2. API:
    1. Python
    2. Matlab
    (3. Commandline interface)
3. 

## Considerations for perfomance 

Compile-time evaluation of Gau�-Laguerre stuff. => pretty hard and no critical path

### Running Matlab on Gitlab

Rough idea:

Execute the Matlab implementation and measure the runtime (and maybe space) of it and compare it with the c++
implementation.

https://stackoverflow.com/questions/34647154/gitlab-ci-with-matlab


## Risks

Precesision of GSL seems to be 2*10^-16, but we need 10^-16
Boost OdeInt seems to be more precise

Boost multiprecision
https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html

About constexpr
https://www.boost.org/doc/libs/1_78_0/libs/multiprecision/doc/html/boost_multiprecision/tut/lits.html

Gau�-laguerre Quadrature with Boost.Multiprecision
https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/fp_eg/gauss_lagerre_quadrature.html

https://stackoverflow.com/questions/37296481/integration-with-quadrature-and-multiprecision-boost-libraries-in-c




## Useful links 


Gaussian quardarature explained:
https://www.youtube.com/watch?v=w2xjlPwYock


About the performanace:
https://stackoverflow.com/questions/18009056/why-does-matlab-octave-wipe-the-floor-with-c-in-eigenvalue-problems

About eigenvalue decomposition (german):
http://www.peter-junglas.de/fh/vorlesungen/numa/html/kap6.html

Running GNU Octave on Gitlab:
https://gitlab.com/mtmiller/octave-snapshot

Comparison of linear algebrar libraries:
https://en.wikipedia.org/wiki/Comparison_of_linear_algebra_libraries


Gnu Scientific library (GSL)

https://www.gnu.org/software/gsl/

Seemingly no constexpr support.

Python API:
https://pybind11.readthedocs.io/en/latest/basics.html
https://github.com/pybind/cmake_example


https://www.boost.org/doc/libs/1_78_0/libs/python/doc/html/index.html

vcpkg and GSL:

https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/

https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/

### CMake
https://cliutils.gitlab.io/modern-cmake/chapters/intro/running.html

### Google results
https://www.google.com/search?channel=trow5&client=firefox-b-d&q=c%2B%2B+library+gauss+laguerre

### Gau�-laguerre implementations

http://www.mymathlib.com/quadrature/gauss_laguerre.html

Gau�-laguerre Quadrature with Boost.Multiprecision
https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/fp_eg/gauss_lagerre_quadrature.html