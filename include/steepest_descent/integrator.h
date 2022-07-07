#pragma once

#include <complex>
#include <functional>

#include "integration/gsl_integrator.h"

namespace integrator
{
	//typedef std::function <const std::complex<double>(const std::complex<double>)> integrand_fun;
	//typedef std::function <const std::complex<double>(const integrand_fun fun, const std::complex<double> first_split_point, const std::complex<double> second_split_point)> integrator;


	template<class T>
	struct integrator {
		integrator() : integrator_impl(T()) {
		}

		template<class integrand_fun>
		auto operator()(integrand_fun integrand, const std::complex<double> first_split_point, const std::complex<double> second_split_point) const ->std::complex<double> {
			return integrator_impl(integrand, first_split_point, second_split_point);
		}

	private:
		T integrator_impl;
	};


	//template<>
	//struct integrator<gsl_integrator> 
	//struct gsl_integrator : integrator<gsl_integrator> {};
	//template<class T> using gsl_integrator = gsl_integration;
	typedef integrator<gsl_integrator> gsl_integration;
}