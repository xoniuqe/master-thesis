// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include <complex> 
#include <variant>
#include <tuple>

#include <armadillo>
#include <steepest_descent/path_utils.h>
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
	integral::integral_1d integral1d(k, &gslintegrator, 0.1, 30);
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
	integral::integral_2d integral2d(k, &gslintegrator, &gsl_integrator_2d, 0.1, 0.1, 30);
	auto res = integral2d(A, b, r, mu);
	std::cout << "result 2d: " << res << std::endl;
	return;
}



int main()
{
	/* 
	// use this as tesT?=
	arma::mat  A{{0, 0}, {1, 0}, {1, 1}};

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };
	arma::vec mu{ 1,-0.5, -0.5 };


	auto k = 10.;
	auto y = 1.;
	auto q = 0.5;
	auto qx = arma::dot(A.col(0), mu);
	auto qy = arma::dot(A.col(1), mu);
	auto prod = arma::dot(mu, b);

	auto [c, c_0] = math_utils::get_complex_roots(y, A, b, r);

	auto spec_point = math_utils::get_spec_point(q, { c, c_0 });
	auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });


	std::cout << "c: " << c << std::endl;
	std::cout << "c_0: " << c_0 << std::endl;

	auto local_k = k;
	auto green_fun_2d = [k = local_k, &A, &b, &r, qx, qy, prod](const double x, const double y) -> auto {
		auto Px = math_utils::calculate_P_x(x, y, A, b, r);
		auto sqrtPx = std::sqrt(Px);
		auto res = std::exp(1.i * k * (sqrtPx + qx * x + qy * y + prod)) * (1. / sqrtPx);
		return res;
	};

	for (auto i = 0.; i < 1.; i += 0.1) {
		//std::cout << "dp(" << i << ") = " << dp(i) << std::endl;
		std::cout << "green_fun_2d(" << i << ", 0) = " << green_fun_2d(i,0) << std::endl;
	}

	integrator::gsl_integrator_2d gsl_integrator_2d;
	auto x = gsl_integrator_2d(green_fun_2d, 0, 1., 0, 0.1);
	std::cout << x << std::endl;*/
	integral_test_1d();

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	integral_test_2d();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	return 0;
}

