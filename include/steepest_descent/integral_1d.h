#pragma once

#include "integrator.h"
#include "datatypes.h"
#include "configuration.h"
#include <armadillo>
namespace integral {


	struct integral_1d {
		integral_1d(const config::configuration config, integrator::gsl_integrator* integrator);
		auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) const -> std::complex<double>;
	private:
		config::configuration config;
		integrator::gsl_integrator* integrator;
		std::vector<double> nodes, weights;

	};

}