#pragma once

#include <vector>
#include <functional>
#include <complex> //include complex to replace the gsl complex numbers
#include <gsl/gsl_complex.h>
#include <gsl/gsl_eigen.h>
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
		auto workspace = gsl_eigen_symmv_alloc(n);
		auto T = gsl_matrix_alloc(n, n);
		gsl_matrix_set_zero(T);
		for (auto i = 0; i < n; i++) {
			gsl_matrix_set(T, i, i, alpha[i]);
			if (i + 1 < n) {
				gsl_matrix_set(T, i, i + 1, beta[i]);
				gsl_matrix_set(T, i + 1, i, beta[i]);
			}
		}
		print_matrix_pretty(T, n, n);
		auto evec = gsl_matrix_alloc(n, n);
		auto laguerrePoints = gsl_vector_alloc(n);
		gsl_matrix_set_zero(evec);
		gsl_vector_set_zero(laguerrePoints);
		auto result = gsl_eigen_symmv(T, laguerrePoints, evec, workspace);
		auto diag = gsl_matrix_diagonal(evec);


		auto quadratureWeights = gsl_vector_alloc(n);
		gsl_vector_set_zero(quadratureWeights);
		for (auto i = 0; i < n; i++) {
			auto value = abs(gsl_matrix_get(evec, 0, i));
			value *= value;
			gsl_vector_set(quadratureWeights, i, value);
		}
	

		auto barycentricWeights = gsl_vector_alloc(n);
		gsl_vector_set_zero(barycentricWeights);
		auto sqrt_of_laguerre_points = gsl_vector_alloc(n);
		gsl_vector_set_zero(sqrt_of_laguerre_points);

		auto tmp = gsl_vector_alloc(n);
		gsl_vector_set_zero(tmp);
		auto max_value = -1.;
		for (auto i = 0; i < n; i++) {
			gsl_vector_set(tmp, i, abs(gsl_matrix_get(evec, 0, i)));
			gsl_vector_set(sqrt_of_laguerre_points, i, sqrt(gsl_vector_get(laguerrePoints, i)));
			gsl_vector_set(barycentricWeights, i, abs(gsl_matrix_get(evec, 0, i)) * sqrt(gsl_vector_get(laguerrePoints, i)));
			auto value = abs(gsl_matrix_get(evec, 0, i)) * sqrt(gsl_vector_get(laguerrePoints, i));

			if (value > max_value) {
				max_value = value;
			}
		}

		gsl_vector_scale(barycentricWeights, 1. / max_value);
		for (auto i = 1; i < n; i += 2) {
			auto value = gsl_vector_get(barycentricWeights, i);
			gsl_vector_set(barycentricWeights, i, -value);
		}

		gsl_eigen_symmv_free(workspace);

		return std::make_tuple(laguerrePoints, quadratureWeights, barycentricWeights);
	}
	
	auto calculate_integral(const path_function, std::vector<double> nodes, std::vector<double> weights) {
		//std::vector<double> calculation_result;
		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), [](auto left, auto right) -> {
			return left * path_function(right);
		}, std::plus<>());
	}
}