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


Needs homebrew to build, and the clang compiler

Install cmake 
brew install cmake

Install ninja
brew install ninja

Install gsl
brew install gsl

Install tbb
brew install tbb

Install armadillo
brew install armadillo


clone the repo
git submodule sync --recursive
git submodule update --init --recursive 


    
### Build with CMAKE

build the project
Build wiht -DSTEDEPY_BUILD_MATLAB=TRUE to build the matlab plugin. This requires Matlab versions >= R2018a

- cmake -S . -B build -DTBB_TEST=FALSE
- cmake --build build -v 

This build the project into out/build/

### CMake options

There are a few custom cmake options that can toggle certain details:
|       option                    | description                                         | default value|
|---------------------------------|-----------------------------------------------------|--------------|
|STEDEPY_TEST                     | Enables the unit tests                              | ON           |
|STEDEPY_BENCHMARK                | Enable benchmarks                                   | ON           |
|STEDEPY_BUILD_MATLAB             | Builds the mex functions                            | OFF          |
|STEDEPY_BUILD_PYTHON_MODULE      | Builds the python module                            | ON           |
|STEDEPY_EXTERN_TBB               | Toggles oneTBB submodule is not build/ loaded       | OFF          |
|STEDEPY_1D_INSTEAD_OF_PARTIAL    | Uses the 1d case instead of a local function        | ON           |
|STEDEPY_USE_DIRECT_STEEPEST_DESC | toggles usage of direct calc. of  stepest_descent   | ON           |

## How to run

### From Visual Studio 2022

On Windows the project and some tests can be run from Visual Studio. If the WSL2 Subsystem with UBUNTU is installed the ubuntu setup can also be debugged.
In the fodler SteepestDescent are some test files.


### Matlab

After the project was build with cmake (and -DSTEDEPY_BUILD_MATLAB=TRUE ), the output folder should contain some compiled mexfiles (build/out/<Target>/api/matlab).

It may be necessary to set add -DMatlab_ROOT_DIR=<root_dir_of_matlab> (example: MACOS -DMatlab_ROOT_DIR=/Application/MATLAB_R2022b.app/)

This files have to loaded in the matlab environment and then the functions can be run by the name of the files (StedepyMatlab_2d and StedepyMatlab_1d)



### Python
After the project was build with cmake , the output folder should contain the compiled  python module (build/out/<Target>/api/python).
Python3 can load this if it is in its search path, or in the current working directory.



## Folder structure
|path             | description                                                     |
|-----------------|-----------------------------------------------------------------|
.\api             | contains sub projects for matlab and python plugins             |
.\benchmark       | the benchmarks with googlebenchmark                             |
.\docs            | some papers  and the main paper                                 |
.\extern          | git submodules                                                  |
.\include         | the header files of the implementation                          |
.\latex           | the latex files for the thesis                                  |
.\src             | the cpp files for the headers                                   |
.\SteepestDescent | a demo implementation with test functions (not production ready)|
.\tests           | catch2 unit tests                                               |
.\vcpkg           | vcpkg submodule for my windows build                            |    

## Implementation details


### Matlab plugin details

Implemented as so-called 'Mexfunctions'. Assumes RAII (explain that concept in a footnote?)
### Used technologies

Programming language: C++

Mathlibrary: Gnu Scientifc Library

Testing: Catch2

Benchmarking: googlebenchmark

Speedup: Intel-Threading-Building-Blocks, OpenCl
