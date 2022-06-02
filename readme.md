# Master thesis


## Setting up my project:

* clone the repo
* cd vcpkg
* bootstrap-vcpkg.bat
* vcpkg install gsl gsl:x64-windows
(todo: vcpkg install @ResponseFile)
* vcpkg integrate install

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

Uses the function f returned by WeightFunPerPath (3) to calculate the actual integral by using the gauﬂ-laguerre quadrature rule. We construct paths along the spitting points Sp_1 and Sp_2.

With f we evaluate the complex function at the nodes calculated by (1) and weigh the results by the matching weights. The result is summed up and is the
wanted integral for the path.

### (3) WeightFunPerPath

Calls GetComplexPath for the given splitting point which returns the path and its derivative. With that we construct a function which evaluates the complex function at the given path.
In essence it calculates f(h(x)) for us, with x = [a,b].


## Implementation details

### Used technologies

Programming language: C++

Mathlibrary: Gnu Scientifc Library

Testing: Catch2

Benchmarking: to be decided

Speedup: Intel-Threading-Building-Blocks, OpenCl, Cuda(?)

### Requirements

1. The code should run faster than the Matlab implementation
2. API:
    1. Python
    2. Matlab
    (3. Commandline interface)
3. 

## Considerations for perfomance 

Compile-time evaluation of Gauﬂ-Laguerre stuff.

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

Gauﬂ-laguerre Quadrature with Boost.Multiprecision
https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/fp_eg/gauss_lagerre_quadrature.html

https://stackoverflow.com/questions/37296481/integration-with-quadrature-and-multiprecision-boost-libraries-in-c




## Useful links 

<<<<<<< Updated upstream
=======
Gaussian quardarature explained:
https://www.youtube.com/watch?v=w2xjlPwYock


About the performanace:
https://stackoverflow.com/questions/18009056/why-does-matlab-octave-wipe-the-floor-with-c-in-eigenvalue-problems

About eigenvalue decomposition (german):
http://www.peter-junglas.de/fh/vorlesungen/numa/html/kap6.html

>>>>>>> Stashed changes
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

### Gauﬂ-laguerre implementations

http://www.mymathlib.com/quadrature/gauss_laguerre.html

Gauﬂ-laguerre Quadrature with Boost.Multiprecision
https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/fp_eg/gauss_lagerre_quadrature.html