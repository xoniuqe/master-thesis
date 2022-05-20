#include <steepest_descent/gauss_laguerre.h>

#include <functional>
#include <complex> //include complex to replace the gsl complex numbers
#include <armadillo>
#include <numeric>
#include <algorithm> 

using namespace std::literals::complex_literals;

namespace gauss_laguerre {

	auto calculate_laguerre_points_and_weights(int n) ->std::tuple<std::vector<double>, std::vector<double>>
	{
		std::vector<double> alpha(n);
		std::iota(std::begin(alpha), std::end(alpha), 1);
		std::for_each(std::begin(alpha), std::end(alpha), [](auto& x) { x = 2. * x - 1.;  });
		std::vector<double> beta(n);
		std::iota(std::begin(beta), std::end(beta), 1);

		arma::mat T(n, n);
		T.zeros();
		for (auto i = 0; i < n; i++) {
			T(i, i) = alpha[i];
			if (i + 1 < n) {
				T(i, i + 1) = beta[i];
				T(i + 1, i) = beta[i];
			}
		}


		arma::mat evec(n, n);
		arma::vec laguerre_points(n);

		auto result = arma::eig_sym(laguerre_points, evec, T);
		auto diag = evec.diag();


		arma::vec quadrature_weights(n);
		for (auto i = 0; i < n; i++) {
			auto value = std::abs(evec(0, i));
			quadrature_weights[i] = value * value;
		}

		arma::vec barycentric_weights(n);
		auto max_value = -1.;
		for (auto i = 0; i < n; i++) {
			barycentric_weights[i] = std::abs(evec(0, i) * std::sqrt(laguerre_points[i]));
			if (barycentric_weights[i] > max_value) {
				max_value = barycentric_weights[i];
			}
		}

		barycentric_weights *= -(1. / max_value);

		return std::make_tuple(arma::conv_to<std::vector<double>>::from(laguerre_points), arma::conv_to<std::vector<double>>::from(quadrature_weights));
	}

	auto calculate_integral(const path_utils::path_function path, std::vector<double> nodes, std::vector<double> weights) -> std::complex<double> {
		//std::vector<double> calculation_result;
		auto result = 0. + 0.i;
		auto reduce_function = [&](auto left, auto right) -> auto {
			return left * path(right);
		};

		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), result, reduce_function, std::plus<>());
	}
}