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
	
	// das ganze funktor object ist fragwürdig, da diese funktion nur einmal mit einem Satz an parametern aufgerufen werdne wird
	//k könnte auch parameter sein
	integral_1d::integral_1d(int k, integrator::integrator integrator, double tolerance) : k(k), integrator(integrator), tolerance(tolerance) {

	}

	 
	// prüfen ob A, b, r und mu überhaupt in dieser Form so benötigt sind
	// im 1d Fall müsste das alles auf ein x zurückfallen?
	// bzw. könnten diese auch in den constructor verscoben werden?
	auto integral_1d::operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu, const double y, const double left_split, const double right_split) const -> std::complex<double> {
		// move this to the outside
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
		if (math_utils::is_within_split_point_tolerances(spec_point, left_split, right_split, tolerance)) {
			split_points = math_utils::get_split_points_spec(q, k, s, { c, c_0 }, left_split, right_split);
		}
		else if (math_utils::is_within_split_point_tolerances(sing_point, left_split, right_split, tolerance)) {
			split_points = math_utils::get_split_points_sing(q, k, s, { c, c_0 }, left_split, right_split);
		} 
		else
		{
			//no singularity
			return steepest_desc(left_split, right_split);
		}

		auto [sp1, sp2] = split_points;

		
		auto I1 = steepest_desc(left_split, sp1);
		auto I2 = steepest_desc(sp2, right_split);


		//just the "normal" integrate is missing
		//    greenFun1D = @(x) exp(1i*k*(Px(x,y,A,b,r).^(1/2)+q*x+s)).*Px(x,y,A,b,r).^(-1/2); 
		//matlab:  integral(greenFun1D,splitPt1,splitPt2, "ArrayValued", "True") 
		auto& local_k = this->k;
		auto green_fun = [k=local_k, y=y, &A, &b, &r, q, s](const std::complex<double> x) -> auto {
			auto Px = math_utils::calculate_P_x(x, y, A, b, r);
			auto sqrtPx = std::sqrt(Px);
			auto res =  std::exp(1.i * k * (sqrtPx + q * x + s)) * (1. / sqrtPx);
			return res;
		};
		auto x = integrator(green_fun, sp1, sp2);
		//std::cout << "result: " << I1 + x + I2 << std::endl;
		return I1 + x + I2;
	}

	integral_2d::integral_2d(const int k, const steepest_descent::steepest_descend_2d steepest_descent_2d, integrator::integrator integrator, const double resolution, const double tolerance) : k(k), integrator(integrator), steepest_descend_2d(steepest_descend_2d), resolution(resolution), tolerance(tolerance) {

	}

	auto integral_2d::operator()(const arma::mat& A, const arma::vec& b, const arma::vec& r, const arma::vec& mu) const->std::complex<double>
	{

	}
}
