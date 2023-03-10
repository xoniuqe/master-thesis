

#include <steepest_descent/complex_comparison.h>
#include <steepest_descent/integral_2d.h>
#include <steepest_descent/integral_1d.h>

#include <steepest_descent/math_utils.h>
#include <steepest_descent/steepest_descent.h>
#include <steepest_descent/gauss_laguerre.h>
#include <steepest_descent/integration/gsl_integrator_2d.h>

#include <armadillo>
#include <cmath>
#include <complex>
#include <mutex>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>


namespace integral {
	using namespace std::complex_literals;

	integral_2d::integral_2d(const config::configuration_2d config, integrator::gsl_integrator* integrator, integrator::gsl_integrator_2d* integrator_2d) : config(config), integrator(integrator), integrator_2d(integrator_2d) {
		auto [n, w] = gauss_laguerre::calculate_laguerre_points_and_weights(config.gauss_laguerre_nodes);
		this->nodes = n;
		this->weights = w;

	}

	integral_2d::integral_2d(const config::configuration_2d config, integrator::gsl_integrator* integrator, integrator::gsl_integrator_2d* integrator_2d, const std::vector<double> nodes, const std::vector<double> weights) : 
		config(config), 
		integrator(integrator), 
		integrator_2d(integrator_2d),
		nodes(nodes),
		weights(weights)
	{


	}



	auto integral_2d::operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& theta) const -> std::complex<double> {


		auto q = arma::dot(A.col(0), theta);
		auto qx = arma::dot(A.col(0), theta);
		auto qy = arma::dot(A.col(1), theta);

		auto prod = arma::dot(theta, b);

		auto A1 = arma::mat(A);
		A1.swap_cols(0, 1);
		auto q1 = arma::dot(A1.col(0), theta);
		auto sx1 = arma::dot(A1.col(1), theta) * 0. + prod; // macht relativ wenig sinn, mal mit paper abgleichen
		auto croots1= math_utils::get_complex_roots(0, A1, b, r);
		auto c1 = std::get<0>(croots1);
		auto c1_0 = std::get<1>(croots1);
		arma::mat rotation{ {-1, 1}, {1, 0} };
		arma::mat A2 = A * rotation;
		auto q2 = arma::dot(A2.col(0), theta);
		auto sx2 = arma::dot(A2.col(1), theta) * 1. + prod; // macht relativ wenig sinn, mal mit paper abgleichen
		auto croots2 = math_utils::get_complex_roots(1, A2, b, r);
		auto c2 = std::get<0>(croots2);
		auto c2_0 = std::get<1>(croots2);

		auto local_k = config.wavenumber_k;
		auto green_fun_2d = [k = local_k, A=A, b=b, r=r, qx=qx, qy=qy, prod=prod](const double x, const double y) -> auto {
			auto Px = math_utils::calculate_P_x(x, y, A, b, r);
			auto sqrtPx = std::sqrt(Px);
			auto res = std::exp(1.i * k * (sqrtPx + qx * x + qy * y + prod)) * (1. / sqrtPx);
			return res;
		};

		auto integration_result = 0. + 0.i;
		int number_of_steps = (int)1. / config.y_resolution;
#ifdef STEDEPY_1D_INSTEAD_OF_PARTIAL
		integrator::gsl_integrator integrator_1d;
		config::configuration config1d;
		config1d.wavenumber_k = config.wavenumber_k;
		config1d.tolerance = config.tolerance;
		math_utils::green_fun_generator fun_gen = [](const double& k, const std::complex<double>& y, const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double& q, const std::complex<double>& s) -> math_utils::green_fun
		{
				return [&](const double x) -> auto {
					auto Px = math_utils::calculate_P_x(x, y, A, b, r);
					auto sqrtPx = std::sqrt(Px);
					auto res = std::exp(1.i * k * (sqrtPx + q * x + s));
					return res;
				};
		};
		integral_1d partial_integral(config1d, &integrator_1d, nodes, weights, path_utils::get_weighted_path_y, fun_gen);
#endif
		integration_result = tbb::parallel_reduce(tbb::blocked_range(0, number_of_steps, 1), 0. + 0.i, [&](tbb::blocked_range<int> range, std::complex<double> integral) {
			integrator::gsl_integrator_2d integrator;
			for (int i = range.begin(); i < range.end(); ++i)
			{
				auto y = config.y_resolution * (double)i;// steps[i];
				auto u = y + config.y_resolution * 0.5;
				auto [c, c_0] = math_utils::get_complex_roots(u, A, b, r);
				auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
				auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

				auto s = arma::dot(A.col(0), theta) * u + prod;

				auto is_spec = math_utils::is_singularity_in_layer(config.tolerance, spec_point, 0, 1 - u);
				auto is_sing = math_utils::is_singularity_in_layer(config.tolerance, sing_point, 0, 1 - u);


#ifdef STEDEPY_1D_INSTEAD_OF_PARTIAL
				auto integration_y = partial_integral(A1, b, r, q1, sx1, 0., c1, c1_0, y, y + config.y_resolution);
				auto integration_1_minus_y = partial_integral(A2, b, r, q2, sx2, 1., c2, c2_0, y, y + config.y_resolution);
#else
				auto integration_y = integrate_lambda(A1, b, r, 0., q1, sx1, c1, c1_0, y, y + config.y_resolution);
				auto integration_1_minus_y = integrate_lambda(A2, b, r, 1., q2, sx2, c2, c2_0, y, y + config.y_resolution);
#endif

#ifndef STEDEPY_USE_DIRECT_STEEPEST_DESC
				steepest_descent::steepest_descend_2d steepest_desc(path_utils::get_weighted_path_2d, nodes, weights, config.wavenumber_k, u, A, b, r, q, s, { c, c_0 }, sing_point);
#endif
				//usage of steepest_descent_2d seems to be slower?
				std::tuple<std::complex<double>, std::complex<double>> split_points;
#ifdef STEDEPY_USE_DIRECT_STEEPEST_DESC
				auto path = path_utils::get_weighted_path_2d(0, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
				auto Iin = gauss_laguerre::calculate_integral_cauchy_tbb(path, nodes, weights);

				auto path2 = path_utils::get_weighted_path_2d(1 - u, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
				auto Ifin = gauss_laguerre::calculate_integral_cauchy_tbb(path2, nodes, weights);
#else
				auto Iin = steepest_desc(0);
				auto Ifin = steepest_desc(1 - u);
#endif
				//Rausgezogen, da es sp???ter sowieso berechnet wird (4.10)
				integral += Iin * integration_y - Ifin * integration_1_minus_y;

				if (!is_spec && !is_sing) {
					// no singularity
					continue;
				}
				if (is_spec) {
					split_points = math_utils::get_split_points_spec(q, config.wavenumber_k, s, { c, c_0 }, 0, 1 - u);
				}
				if (is_sing) {
					split_points = math_utils::get_split_points_sing(q, config.wavenumber_k, s, { c, c_0 }, 0, 1 - u);
				}
				auto& [split_point1, split_point2] = split_points;


				//singularity
#ifdef STEDEPY_USE_DIRECT_STEEPEST_DESC
				auto path3 = path_utils::get_weighted_path_2d(split_point1, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
				auto Ifin1 = gauss_laguerre::calculate_integral_cauchy_tbb(path3, nodes, weights);

				auto path4 = path_utils::get_weighted_path_2d(split_point2, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
				auto Iin2 = gauss_laguerre::calculate_integral_cauchy_tbb(path4, nodes, weights);
#else

				auto Ifin1 = steepest_desc(split_point1);
				auto Iin2 = steepest_desc(split_point2);
#endif

				//Vertical layer at splitPt1.

				auto sx_intern_1 = arma::dot(A1.col(1), theta) * split_point1 + prod;
				auto internRoots1 = math_utils::get_complex_roots(split_point1, A1, b, r);
				auto cIntern1 = std::get<0>(internRoots1);
				auto c_0Intern1 = std::get<1>(internRoots1);
				//Vertical layer at splitPt2.
				auto sx_intern_2 = arma::dot(A1.col(1), theta) * split_point2 + prod;
				auto internRoots2 = math_utils::get_complex_roots(split_point2, A1, b, r);
				auto cIntern2 = std::get<0>(internRoots2);
				auto c_0Intern2 = std::get<1>(internRoots2);


				// Integration over these two segments.
#ifdef STEDEPY_1D_INSTEAD_OF_PARTIAL				
				auto intYintern1 = partial_integral(A1, b, r, q1, sx_intern_1, split_point1, cIntern1, c_0Intern1, y, y + config.y_resolution);
				auto intYintern2 = partial_integral(A1, b, r, q1, sx_intern_2, split_point2, cIntern2, c_0Intern2, y, y + config.y_resolution);
#else
				auto intYintern1 = integrate_lambda(A1, b, r, split_point1, q1, sx_intern_1, cIntern1, c_0Intern1, y, y + config.y_resolution);
				auto intYintern2 = integrate_lambda(A1, b, r, split_point2, q1, sx_intern_2, cIntern2, c_0Intern2, y, y + config.y_resolution);
#endif

				auto integral2_res = integrator.operator()(green_fun_2d, split_point1, split_point2, y, y + config.y_resolution);
				//auto integral2_res = integrator.operator()(green_fun_2d, 0., 1., 0.01, 0.02);// +config.y_resolution);

				auto integration_result = integral2_res + Iin2 * intYintern2 - Ifin1 * intYintern1;
		
				//auto integration_result = integral2_res;
				integral += integration_result;
			}
			return integral;
			}, std::plus<std::complex<double>>());
		return integration_result;
	}


	auto integral_2d::integrate_lambda(const arma::mat& A, const arma::vec& b, const arma::vec& r, const std::complex<double> sPx, const double q, const std::complex<double> s, const std::complex<double> c, const double c_0, const double left_split, const double right_split) const -> std::complex<double> {
		auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
		auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

		if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) {
			sing_point = std::real(sing_point);
		}

		if (std::abs(std::imag(spec_point)) < std::abs(std::imag(c))) {
			spec_point = std::real(spec_point);
		}

		steepest_descent::steepest_descend_2d steepest_desc(path_utils::get_weighted_path_y, nodes, weights, config.wavenumber_k, sPx, A, b, r, q, s, { c, c_0 }, sing_point);


		auto is_spec = math_utils::is_singularity_in_layer(config.tolerance, spec_point, left_split, right_split);
		auto is_sing = math_utils::is_singularity_in_layer(config.tolerance, sing_point, left_split, right_split);

		std::tuple<std::complex<double>, std::complex<double>> split_points;
		if (!is_sing && !is_spec)
		{
			auto left = steepest_desc(left_split);
			auto right = steepest_desc(right_split);
			//no singularity
			return left - right;
		}
		if (is_spec) {
			split_points = math_utils::get_split_points_spec(q, config.wavenumber_k, s, { c, c_0 }, left_split, right_split);
		}
		if (is_sing) {
			split_points = math_utils::get_split_points_sing(q, config.wavenumber_k, s, { c, c_0 }, left_split, right_split);
		}
    

		auto& [sp1, sp2] = split_points;

		// -> optimization: if either of these checks is true the according integral will be zero: 4.10.22 hat fast 200 ms gebracht in benchmarks!
		auto I1 = std::abs(left_split - sp1) <=  std::numeric_limits<double>::epsilon() ? 0. : steepest_desc(left_split) - steepest_desc(sp1);

		auto I2 = std::abs(sp2 - right_split) <= std::numeric_limits<double>::epsilon() ? 0. :  steepest_desc(sp2) - steepest_desc(right_split);

		auto green_fun = [k = config.wavenumber_k, &sPx, &A, &b, &r, q, s](const double x) -> auto {
			auto Px = math_utils::calculate_P_x(x, sPx, A, b, r);
			auto sqrtPx = std::sqrt(Px);
			auto res = std::exp(1.i * k * (sqrtPx + q * x + s)); //this formular differs because equation 28 in PAPERHIFE!
			return res;
		};
		auto x = integrator->operator()(green_fun, sp1, sp2);
		return I1 + x + I2;
	}

}
