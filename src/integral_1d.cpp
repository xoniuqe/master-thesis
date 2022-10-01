#include <steepest_descent/integral_1d.h>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/steepest_descent.h>
#include <steepest_descent/gauss_laguerre.h>

#include <armadillo>
#include <cmath>
#include <complex>


#ifdef _WIN32
//#include <corecrt_math_defines.h>
#endif

namespace integral {
	using namespace std::complex_literals;

	integral_1d::integral_1d(const config::configuration config, integrator::gsl_integrator* integrator) : config(config), integrator(integrator) {
		auto [n, w] = gauss_laguerre::calculate_laguerre_points_and_weights(config.gauss_laguerre_nodes);
		this->nodes = n;
		this->weights = w;
	}


	auto integral_1d::operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const arma::vec3& theta, const double y, const double left_split, const double right_split) const -> std::complex<double> {
		return integral_1d::operator()(A, b, r, arma::dot(A.col(0), theta), arma::dot(A.col(1), theta) * y + arma::dot(theta, b), y, left_split,right_split);
	}


	auto integral_1d::operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double q, const double s, const double y, const double left_split, const double right_split) const -> std::complex<double> {
		auto [c, c_0] = math_utils::get_complex_roots(y, A, b, r);

		auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
		auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

		if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) {
			sing_point = std::real(sing_point);
		}

		if (std::abs(std::imag(spec_point)) < std::abs(std::imag(c))) {
			spec_point = std::real(spec_point);
		}
		steepest_descent::steepest_descend_1d steepest_desc(nodes, weights, config.wavenumber_k, y, A, b, r, q, s, { c, c_0 }, sing_point);

		std::tuple<std::complex<double>, std::complex<double>> split_points;
		if (math_utils::is_singularity_in_layer(config.tolerance, spec_point, left_split, right_split)) {
			split_points = math_utils::get_split_points_spec(q, config.wavenumber_k, s, { c, c_0 }, left_split, right_split);
		}
		else if (math_utils::is_singularity_in_layer(config.tolerance, sing_point, left_split, right_split)) {
			split_points = math_utils::get_split_points_sing(q, config.wavenumber_k, s, { c, c_0 }, left_split, right_split);
		} 
		else
		{
			//no singularity
			return steepest_desc(left_split, right_split);

		}

		auto& [sp1, sp2] = split_points;


		auto I1 = steepest_desc(left_split, sp1);

		auto I2 =  steepest_desc(sp2, right_split);

		auto& local_k = this->config.wavenumber_k;
		auto green_fun = [k=local_k, y=y, &A, &b, &r, q, s](const double x) -> auto { 
			auto Px = math_utils::calculate_P_x(x, y, A, b, r);
			auto sqrtPx = std::sqrt(Px);
			auto res =  std::exp(1.i * k * (sqrtPx + q * x + s)) * (1. / sqrtPx);
			return res;
		};
		auto x = integrator->operator()(green_fun, sp1, sp2);
		return I1 + x + I2;
	}


}
