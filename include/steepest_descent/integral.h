#pragma once

#include "integrator.h"
#include "datatypes.h"
#include <armadillo>

namespace integral {


	//auto integral_1d(const int k, const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) -> auto;

	//auto get_integral_1d(const double y, const triangle_parameter& triangle, const datatypes::vector& r,) -> auto;

	struct integral_1d {
		integral_1d(int k, integrator::gsl_integrator integrator, double tolerance);
		auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) const -> std::complex<double>;
	private:
		double k;
		integrator::gsl_integrator integrator;
		double tolerance;

		//auto calculate_paths() -> auto;

	};

}