// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include <complex> 
#include <variant>
#include <tuple>

#include <armadillo>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/integrator.h>
#include <steepest_descent/integral_1d.h>
#include <steepest_descent/integral_2d.h>

#include <type_traits>

using namespace std::complex_literals;



void setup_1d_test()
{
	arma::mat  A{ {-5, 1}, {-1, 1,} , {- 1 ,0 }};
	
	arma::vec b { 0,1,0 };

	arma::vec r { 0, 12, 1 };
	
	arma::vec mu { 1,4,0 };

	auto DPx = math_utils::partial_derivative_P_x(0, 0, A, b, r);


	std::cout << "\nDPx = \n" << DPx;

	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

	std::cout << "\nc = \n" << c;
	std::cout << "\nc0 = \n" << c_0;
	std::cout << std::endl;
}

void test_split() {
	arma::mat  A{ {1, 1}, {1, 1,} , {0 ,0 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 0, 1, 2 };

	arma::vec mu{ 1,4,0 };

	std::cout << "A:\n";
	A.print();

	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

	auto q = -8;
	auto k = 10;
	auto s = 3;

	auto spec_point = math_utils::get_spec_point(q, { c, c_0 });
	auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });

	std::cout << "c: " << c << std::endl;
	std::cout << "c_0: " << c_0 << std::endl;
	std::cout << "spec_point: " << spec_point << std::endl;
	std::cout << "sing_point: " << sing_point << std::endl;

}



auto integral_test_1d()
{
	arma::mat  A{ {0, 0}, {1, 0}, {1, 1 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };

	arma::vec mu{ 1,-0.5, -0.5 };
	auto y = 0.;
	auto k = 100;
	auto left_split = 0.;
	auto right_split = 1.;
	integrator::gsl_integrator gslintegrator;
	integral::integral_1d integral1d(k, &gslintegrator, 0.1);
	auto res = integral1d(A, b, r, mu, y, left_split, right_split);
	std::cout << "result 1d: " << res << std::endl;
	return;
}

auto integral_test_2d() {
	arma::mat  A{ {0, 0}, {1, 0}, {1, 1 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };

	arma::vec mu{ 1,-0.5, -0.5 };
	auto k = 10;

	integrator::gsl_integrator gslintegrator;	
	integrator::gsl_integrator_2d gsl_integrator_2d;
	integral::integral_2d integral2d(k, &gslintegrator, &gsl_integrator_2d, 0.1, 0.1);
	auto res = integral2d(A, b, r, mu);
	std::cout << "result 2d: " << res << std::endl;
	return;
}

auto integral_templated_test() {
	arma::mat  A{ {0, 0}, {1, 0}, {1, 1 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };

	arma::vec mu{ 1,-0.5, -0.5 };
	auto y = 0.;
	auto k = 100;
	auto left_split = 0.;
	auto right_split = 1.;

	/*auto green_fun = [k = k, y = y, &A, &b, &r, q, s](const double x) -> auto { //const std::complex<double> x) -> auto {
		auto Px = math_utils::calculate_P_x(x, y, A, b, r);
		auto sqrtPx = std::sqrt(Px);
		auto res = std::exp(1.i * k * (sqrtPx + q * x + s)) * (1. / sqrtPx);
		return res;
	};

	integral::integral_1d_test<green_fun, steepest_descend_test<path_utils::get_weighted_path_y>> test;*/
}



int main()
{
	//integral_test_1d();
	integral_test_2d();

	return 0;
}

