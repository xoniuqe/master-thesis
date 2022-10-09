#include <steepest_descent/gauss_laguerre.h>

#include <functional>
#include <complex> //include complex to replace the gsl complex numbers
#include <armadillo>
#include <numeric>
#include <algorithm> 
//slower: #define USE_TR


namespace gauss_laguerre {
	using namespace std::literals::complex_literals;


	auto calculate_laguerre_points_and_weights(size_t n) ->std::tuple<std::vector<double>, std::vector<double>>
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
		laguerre_points.zeros();
		evec.zeros();
		arma::eig_sym(laguerre_points, evec, T, "std");
		
		auto diag = evec.diag();
		//differs from matlab


		auto weight_sum = 0.;
		arma::vec quadrature_weights(n);
		for (auto i = 0; i < n; i++) {
			auto value = evec(0, i);
			value = std::pow(value, 2);
			weight_sum += value;
			quadrature_weights[i] = value;
		}

		// normalize quadrature weights
		quadrature_weights *= (1. / weight_sum);


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

	/*auto calculate_integral_cauchy(const path_utils::path_function& path, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double> {
		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), 0. + 0.i, std::plus<std::complex<double>>(), [&](const auto& left, const auto& right) -> auto {
			//slower: return right * (first_path(left) - second_path(left));
			return (right * path(left));
			});*/

		/*std::vector<std::complex<double>> eval_points;
		std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points), path);
		return std::inner_product(weights.begin(), weights.end(), eval_points.begin(), 0. + 0.i);*/
	//}

	auto calculate_integral_cauchy(const path_utils::path_function first_path, const path_utils::path_function second_path, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double> {
#ifdef USE_TR
		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), 0. + 0.i, std::plus<std::complex<double>>(), [&](const auto left, const auto right) -> auto {
			//slower: return right * (first_path(left) - second_path(left));
			return (right * first_path(left) - right * second_path(left));
			});
#else
		std::vector<std::complex<double>> eval_points1, eval_points2, eval_points_summed;
		std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points1), first_path);
		std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points2), second_path);
		std::transform(eval_points1.begin(), eval_points1.end(), eval_points2.begin(), std::back_inserter(eval_points_summed), [](const auto left, const auto right) -> auto {
			return left - right;
			});
		return std::inner_product(weights.begin(), weights.end(), eval_points_summed.begin(), 0. + 0.i);
#endif
	}


}