#pragma once

#include "integration/gsl_integrator.h"
#include "datatypes.h"
#include "configuration.h"
#include "steepest_descent.h"
#include <armadillo>

namespace integral {
	using namespace std::complex_literals;

	struct integral_1d {
		integral_1d(const config::configuration config, integrator::gsl_integrator* integrator);
		integral_1d(const config::configuration config, integrator::gsl_integrator* integrator, const std::vector<double> nodes, const std::vector<double> weights);


		auto operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const arma::vec3& theta, const double y, const double left_split, const double right_split) const->std::complex<double>;
		auto operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double q, const double s, const double y, const double left_split, const double right_split) const -> std::complex<double>;
			
	private:
		config::configuration config;
		integrator::gsl_integrator* integrator;
		std::vector<double> nodes, weights;

		
	};
	
	struct integral_1d_test {

		integral_1d_test(const config::configuration& config, integrator::gsl_integrator* integrator, const std::vector<double> nodes, const std::vector<double> weights) : config(config), integrator(integrator), nodes(nodes), weights(weights) {

		}
		

		auto operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double q, const  std::complex<double> s, const  std::complex<double> y, const double left_split, const double right_split) const->std::complex<double>
		{
			auto [c, c_0] = math_utils::get_complex_roots(y, A, b, r);
			return operator()(A, b, r, q, s, y, c, c_0, left_split, right_split);
			
		}
		inline auto operator()( const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double q, const  std::complex<double> s, const  std::complex<double> y, const std::complex<double> c, const double c_0, const double left_split, const double right_split) const->std::complex<double>
		{
			auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
			auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

			if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) {
				sing_point = std::real(sing_point);
			}

			if (std::abs(std::imag(spec_point)) < std::abs(std::imag(c))) {
				spec_point = std::real(spec_point);
			}
			steepest_descent::steepest_descend_2d steepest_desc(path_utils::get_weighted_path, nodes, weights, config.wavenumber_k, y, A, b, r, q, s, { c, c_0 }, sing_point);
			//steepest_descent::steepest_descend_1d steepest_desc(nodes, weights, config.wavenumber_k, y, A, b, r, q, s, { c, c_0 }, sing_point);

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
				return steepest_desc(left_split) - steepest_desc(right_split);

			}

			auto& [sp1, sp2] = split_points;

			auto I1 = steepest_desc(left_split) - steepest_desc(sp1);

			auto I2 = steepest_desc(sp2) - steepest_desc(right_split);

			auto& local_k = this->config.wavenumber_k;

			//auto fun = green_fun(A, b, r, y, q, s);
			auto fun = [k = config.wavenumber_k, &y, &A, &b, &r, q, s](const double x) -> auto {
				auto Px = math_utils::calculate_P_x(x, y, A, b, r);
				auto sqrtPx = std::sqrt(Px);
				auto res = std::exp(1.i * k * (sqrtPx + q * x + s)); //this formular differs because equation 28 in PAPERHIFE!
				return res;
			};
			auto x = integrator->operator()(fun, sp1, sp2);
			return I1 + x + I2;
		}

	private:
		config::configuration config;
		const integrator::gsl_integrator* integrator;
		const std::vector<double> nodes, weights;


	};
}