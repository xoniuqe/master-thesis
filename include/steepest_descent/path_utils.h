#pragma once

#include "math_utils.h"
#include <complex>

namespace path_utils {

    using namespace std::literals::complex_literals;
    typedef std::function <const std::complex<double>(const std::complex<double> x)> path_function;


	auto generate_K_x(const double x, const double P_x, const double q)->std::function<const std::complex<double>(const double t)>;


	auto get_complex_path(const double split_point, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, const double q, const datatypes::complex_root complex_root)->std::tuple<path_function, path_function>;

}