#pragma once

#include "integration/gsl_integrator.h"
#include "integration/gsl_integrator_2d.h"
#include "datatypes.h"
#include "configuration.h"
#include <armadillo>
#include <vector>
#include <complex>

namespace integral {

	struct integral_2d {
		integral_2d(const config::configuration_2d config, integrator::gsl_integrator* integrator, integrator::gsl_integrator_2d* integrator_2d);
		integral_2d(const config::configuration_2d config, integrator::gsl_integrator* integrator, integrator::gsl_integrator_2d* integrator_2d, const std::vector<double> nodes, const std::vector<double> weights);
		
        auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& theta) const->std::complex<double>;	
	private:
		config::configuration_2d config;
		integrator::gsl_integrator* integrator;
		integrator::gsl_integrator_2d* integrator_2d;
		std::vector<double> nodes, weights;

		auto get_partial_integral(const arma::mat& A, const arma::vec& b, const arma::vec& r, const std::complex<double> sPx, const double q,  const std::complex<double> s, const std::complex<double> c, const double c_0, const double left_split, const double right_split) const ->std::complex<double>;
	};
}