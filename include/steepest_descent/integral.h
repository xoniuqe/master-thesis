#pragma once

#include "integrator.h"
#include "datatypes.h"
#include "steepest_descent.h"
#include <armadillo>

namespace integral {


	//auto integral_1d(const int k, const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) -> auto;

	//auto get_integral_1d(const double y, const triangle_parameter& triangle, const datatypes::vector& r,) -> auto;

	struct integral_1d {
		integral_1d(int k, integrator::integrator integrator, double tolerance);
		auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) const -> std::complex<double>;
	private:
		double k;
		//steepest_descent::steepest_descend_1d steepest_descen;
		integrator::integrator integrator;
		double tolerance;

		//auto calculate_paths() -> auto;

	};

	struct integral_2d {
		integral_2d(const int k, const steepest_descent::steepest_descend_2d steepest_descent_2d,const integrator::integrator integrator, const double resolution, const double tolerance);
		auto operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu) const->std::complex<double>;
	private:
		double k;
		//steepest_descent::steepest_descend_1d;
		steepest_descent::steepest_descend_2d steepest_descend_2d;
		integrator::integrator integrator;
		double tolerance, resolution;
	};

}