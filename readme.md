# Master thesis


## Setting up my project:

* clone the repo
* cd vcpkg
* bootstrap-vcpkg.bat
* vcpkg install gsl gsl:x64-windows
(todo: vcpkg install @ResponseFile)
* vcpkg integrate install

## Implementation details

### Used technologies

Programming language: C++

Mathlibrary: Gnu Scientifc Library

Testing: do be decided

Speedup: Intel-Threading-Building-Blocks, OpenCl, Cuda(?)

### Requirements

1. The code should run faster than the Matlab implementation
2. API:
    1. Python
    2. Matlab
    3. Commandline interface
3. 

## Considerations for perfomance 

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

Gauﬂ-laguerre Quadrature with Boost.Multiprecision
https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/fp_eg/gauss_lagerre_quadrature.html

https://stackoverflow.com/questions/37296481/integration-with-quadrature-and-multiprecision-boost-libraries-in-c


## Useful links 


Comparison of linear algebrar libraries:
https://en.wikipedia.org/wiki/Comparison_of_linear_algebra_libraries


Gnu Scientific library (GSL)

https://www.gnu.org/software/gsl/


vcpkg and GSL:

https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/

https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/

### CMake
https://cliutils.gitlab.io/modern-cmake/chapters/intro/running.html

### Google results
https://www.google.com/search?channel=trow5&client=firefox-b-d&q=c%2B%2B+library+gauss+laguerre