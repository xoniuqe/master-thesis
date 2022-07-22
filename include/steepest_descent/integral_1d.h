#pragma once

#include "integrator.h"
#include "datatypes.h"
#include <armadillo>
namespace integral {


	struct integral_1d {
		integral_1d(int k, integrator::gsl_integrator* integrator, double tolerance);
		auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) const -> std::complex<double>;
	private:
		double k;
		integrator::gsl_integrator* integrator;
		double tolerance;

	};

}