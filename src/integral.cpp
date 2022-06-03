


#include <steepest_descent/integral.h>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/steepest_descent.h>
#include <steepest_descent/gauss_laguerre.h>

#include <armadillo>
#include <cmath>
#include <complex>


#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif

namespace integral {
	using namespace std::complex_literals;

	integral_1d::integral_1d(int k, integrator::integrator integrator, double tolerance) : k(k), integrator(integrator), tolerance(tolerance) {

	}



	auto integral_1d::operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) const -> std::complex<double> {
		auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(30);

		auto q = arma::dot(A.col(0), mu);
		auto s = arma::dot(A.col(1), mu) * y + arma::dot(mu, b);
		auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

		auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
		auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

		if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) {
			sing_point = std::real(sing_point);
		}

		if (std::abs(std::imag(spec_point) < std::abs(std::imag(c)))) {
			spec_point = std::real(spec_point);
		}
		steepest_descent::steepest_descend_1d steepest_desc(nodes, weights, k, y, A, b, r, q, s, { c, c_0 }, sing_point);



		std::tuple<std::complex<double>, std::complex<double>> split_points;
		if (std::real(spec_point) >= (left_split - tolerance) && std::real(spec_point) <= (right_split + tolerance) && std::abs(std::imag(spec_point)) <= std::numeric_limits<double>::epsilon()) {
			std::cout << "spec point" << spec_point << std::endl;
			split_points = math_utils::get_split_points_spec(q, k, s, { c, c_0 }, left_split, right_split);
		}
		else if (std::real(sing_point) >= (left_split - tolerance) && std::real(sing_point) <= (right_split + tolerance) && std::abs(std::imag(sing_point)) <= std::numeric_limits<double>::epsilon()) {
			std::cout << "sing point" << sing_point << std::endl;

			split_points = math_utils::get_split_points_sing(q, k, s, { c, c_0 }, left_split, right_split);
		} 
		else
		{
			//no singularity
			return steepest_desc(left_split, right_split);

		}

		auto [sp1, sp2] = split_points;

		
/*		auto path1 = path_utils::get_weighted_path(left_split, y, A, b, r, q, k, s, {c, c_0}, sing_point);
		auto path2 = path_utils::get_weighted_path(sp1, y, A, b, r, q, k, s, { c, c_0 }, sing_point);

		auto I1 = gauss_laguerre::calculate_integral_cauchy(path1, path2, nodes, weights);*/

		auto I1 = steepest_desc(left_split, sp1);

/*
		auto path3 = path_utils::get_weighted_path(sp2, y, A, b, r, q, k, s, { c, c_0 }, sing_point);
		std::cout << "weighted path: " << path3(1) << std::endl;
		auto path4 = path_utils::get_weighted_path(right_split, y, A, b, r, q, k, s, { c, c_0 }, sing_point);
		std::cout << "weighted path: " << path4(1) << std::endl;

		sp1
		auto I2 = gauss_laguerre::calculate_integral_cauchy(path3, path4, nodes, weights);

	*/
		auto I2 =  steepest_desc(sp2, right_split);
		//std::cout << "I1: " << I1 << std::endl;
		//std::cout << "I2: " << I2 << std::endl;

		//just the "normal" integrate is missing
		//    greenFun1D = @(x) exp(1i*k*(Px(x,y,A,b,r).^(1/2)+q*x+s)).*Px(x,y,A,b,r).^(-1/2); 
		//matlab:  integral(greenFun1D,splitPt1,splitPt2, "ArrayValued", "True") 
		auto green_fun = [k=k, y, A, b, r, q, s](const std::complex<double> x) -> auto {
			return std::exp(1.i * k * std::sqrt(math_utils::calculate_P_x(x, y, A, b, r)) + q * x + s) * std::sqrt(math_utils::calculate_P_x(x, y, A, b, r));
		};
		auto x = integrator(green_fun, sp1, sp2);
		//std::cout << "result: " << I1 + x + I2 << std::endl;
		return I1 + x + I2;
	}


}
