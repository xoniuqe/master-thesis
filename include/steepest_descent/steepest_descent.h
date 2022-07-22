#pragma once

#include "datatypes.h"
#include <complex>
#include "gauss_laguerre.h"

namespace steepest_descent {

	struct steepest_descend_1d {
		steepest_descend_1d(std::vector<double> nodes, std::vector<double> weights, const double k, const double y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r,
			const double q, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point);

		auto operator()(const std::complex<double> first_split, const std::complex<double> second_split) const ->std::complex<double>;
	private:
		std::vector<double> nodes, weights;
		double k, y, q, s;
		datatypes::matrix A;
		arma::vec3 b, r;
		datatypes::complex_root complex_root; 
		std::complex<double> sing_point;
	};

	template<typename T>
	struct steepest_descend_2d {
		steepest_descend_2d(T& path_weighter, std::vector<double> nodes, std::vector<double> weights, const double k, const std::complex<double> y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r,
			const double q, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point) : path_weighter(path_weighter), nodes(nodes), weights(weights), k(k), y(y), A(A), b(b), r(r), q(q), s(s), complex_root(complex_root), sing_point(sing_point) {

		}

		auto operator()(const std::complex<double> split_point) const->std::complex<double> {
			auto path = path_weighter(split_point, y, A, b, r, q, k, s, complex_root, sing_point);

			return gauss_laguerre::calculate_integral_cauchy(path, nodes, weights);
		}



	private:
		T& path_weighter;
		std::vector<double> nodes, weights;
		double k, q;
		std::complex<double> y, s;
		datatypes::matrix A;
		arma::vec3 b, r;
		datatypes::complex_root complex_root;
		std::complex<double> sing_point;
	};

}