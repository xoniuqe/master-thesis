#pragma once

#include "integrator.h"
#include "datatypes.h"
#include <armadillo>
#include <vector>
#include <complex>

namespace integral {

	struct integral_2d {
		integral_2d(int k, integrator::gsl_integrator* integrator, integrator::gsl_integrator_2d* integrator_2d, double tolerance, double resolution);
		auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mus) const->std::complex<double>;
	private:
		double k;
		integrator::gsl_integrator* integrator;
		integrator::gsl_integrator_2d* integrator_2d;
		double tolerance, resolution;
		std::vector<double> nodes, weights;

		auto get_partial_integral(const arma::mat& A, const arma::vec& b, const arma::vec& r, const std::complex<double> sPx, const double q,  const std::complex<double> s, const std::complex<double> c, const double c_0, const double left_split, const double right_split) const ->std::complex<double>;

	};
}