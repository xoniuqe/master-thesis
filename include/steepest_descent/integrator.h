#pragma once

#include <complex>
#include <functional>

namespace integrator
{
	typedef std::function <const std::complex<double>(const std::complex<double>)> integrand_fun;
	typedef std::function <const std::complex<double>(const integrand_fun fun, const std::complex<double> first_split_point, const std::complex<double> second_split_point)> integrator;

}