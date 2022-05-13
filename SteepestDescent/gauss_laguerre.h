#pragma once

#include <vector>
#include <functional>
#include <complex> //include complex to replace the gsl complex numbers
#include <armadillo>
#include <tuple>
#include <numeric>
#include <algorithm>

namespace gauss_laguerre {
	//alternative to try: 
	//typedef std::function <const double(const std::complex<double> x)> path_function;

	typedef std::function <const double(const gsl_complex x)> path_function;

	auto calculate_laguerre_points_and_weights(int n) {
		std::vector<double> alpha(n);
		std::iota(std::begin(alpha), std::end(alpha), 1);
		std::for_each(std::begin(alpha), std::end(alpha), [](auto& x) { x = 2. * x - 1.;  });
		std::vector<double> beta(n);
		std::iota(std::begin(beta), std::end(beta), 1);

		arma::mat T(n, n);
		auto workspace = gsl_eigen_symmv_alloc(n);
		auto T = gsl_matrix_alloc(n, n);
		for (auto i = 0; i < n; i++) {
			T[i, i] = alpha[i];
			if (i + 1 < n) {
				T[i, i + 1] = beta[i];
				T[i + 1, i] = beta[i];
			}
		}


		arma::mat evec(n, n);
		arma::vec laguerre_points(n);

		auto result = arma::eig_sym(laguerre_points, evec, T);
		auto diag = evec.diag();


		arma::vec quadrature_weights(n);
		for (auto i = 0; i < n; i++) {
			auto value = abs(evec[0,i]);
			quadrature_weights[i] = value * value;
		}
	

		arma::vec barycentric_weights (n);

		auto max_value = -1.;
		for (auto i = 0; i < n; i++) {
			barycentric_weights[i] = std::abs(evec[0, i] * sqrt(laguerre_points[i]));
			if (barycentric_weights[i] > max_value) {
				max_value = barycentric_weights[i];
			}
		}

		barycentric_weights *= -/1. / max_value);
		return std::make_tuple(laguerre_points, quadrature_weights, barycentric_weights);
	}
	
	auto calculate_integral(const path_function, std::vector<double> nodes, std::vector<double> weights) {
		//std::vector<double> calculation_result;
		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), [](auto left, auto right) -> {
			return left * path_function(right);
		}, std::plus<>());
	}
}