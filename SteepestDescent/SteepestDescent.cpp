// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include <complex> 
#include <variant>
#include <tuple>

#include <armadillo>
#include <steepest_descent/path_utils.h>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/integral_1d.h>
#include <steepest_descent/integral_2d.h>

#include <type_traits>

#include <numeric>

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
	auto left_split = 0.;
	auto right_split = 1.;
	config::configuration config;
	config.wavenumber_k = 100;
	config.tolerance = 0.1;
	config.gauss_laguerre_nodes = 30;
	integrator::gsl_integrator gslintegrator;
	integral::integral_1d integral1d(config, &gslintegrator);
	auto res = integral1d(A, b, r, mu, y, left_split, right_split);
	std::cout << "result 1d: " << res << std::endl;
	return;
}

auto integral_test_2d(double resolution) {
	//arma::mat  A{ {0, 0}, {1, 0}, {1, 1 } };

	//arma::vec b{ 0,0,0 };

	//arma::vec r{ 1, -0.5, -0.5 };

	//arma::vec mu{ 1,-0.5, -0.5 };
	// 
	arma::mat  A{ {1, 1}, {1,1}, {0, 1 } };

	arma::vec b{ 1,0,-1 };

	arma::vec r{ 0, 1, 2};

	arma::vec mu{ 1,0,0 };


	config::configuration_2d config;
	config.wavenumber_k = 10;
	config.tolerance = 0.1;
	config.y_resolution = resolution;
	config.gauss_laguerre_nodes = 30;

	integrator::gsl_integrator gslintegrator;	
	integrator::gsl_integrator_2d gsl_integrator_2d;
	integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);
	auto res = integral2d(A, b, r, mu);
	std::cout << "result 2d: " << res << std::endl;
	return;
}

auto eval_2d_article(int k)
{
	arma::mat  A{ {0, 0}, {1, 0}, {0, 1 } };

	arma::vec b{ 0,0,0 };

//	arma::vec r{ 1, 0, 1 };

	//arma::vec mu{ 1, 4, 0 };
	//auto k = 10;
	config::configuration_2d config;
	config.wavenumber_k = 5;
	config.tolerance = 0.1;
	config.y_resolution = 0.1;
	config.gauss_laguerre_nodes = 600;

	integrator::gsl_integrator gslintegrator;
	integrator::gsl_integrator_2d gsl_integrator_2d;
	integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);

	std::vector<int64_t> timings(40);

	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);

	for (auto i = 0; i < 40; i++) {

		arma::vec3 r { 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };
		arma::vec3 mu{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };


		auto start = std::chrono::steady_clock::now();
		auto res = integral2d(A, b, r, mu);
		auto end = std::chrono::steady_clock::now();;
		timings[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	}
	auto [min, max] = std::minmax_element(timings.begin(), timings.end());
	std::cout << "highest: " << *max << "\n";
	std::cout << "lowest: " << *min << "\n";


	auto average = std::accumulate(timings.begin(), timings.end(), 0.) / 40.;
	std::cout << "Timing K (" << k << ") = " << average << std::endl;

	//std::cout << "Partialintegration details: \nnumber of partial integrations: " << integral2d.partial_integrations << "\nno singularity paths: " << integral2d.partial_no_singularities << "\nfirst resets: " << integral2d.partial_first_reset << "\nsecond resets: " << integral2d.partial_second_reset << std::endl;
}

auto check_accuracy() {
	arma::mat  A{ {0, 0}, {2, 0}, {0, 2} };

	arma::vec b{ 0,-0.5,0 };

	arma::vec r{ 0.0618, 0.1902, 0 };

	arma::vec theta{ 1,0, 0 };


	config::configuration_2d config;
	config.wavenumber_k = 100;
	config.tolerance = 0.1;
	config.y_resolution = 0.0001;
	config.gauss_laguerre_nodes = 1000;

	auto q = arma::dot(A.col(0), theta);
	auto qx = arma::dot(A.col(0), theta);
	auto qy = arma::dot(A.col(1), theta);

	auto prod = arma::dot(theta, b);

	auto A1 = arma::mat(A);
	A1.swap_cols(0, 1);
	auto q1 = arma::dot(A1.col(0), theta);
	auto sx1 = arma::dot(A1.col(1), theta) * 0. + prod; // macht relativ wenig sinn, mal mit paper abgleichen
	auto croots1 = math_utils::get_complex_roots(1, A1, b, r);
	auto c1 = std::get<0>(croots1);
	auto c1_0 = std::get<1>(croots1);

	arma::mat rotation{ {-1, 1}, {1, 0} };
	arma::mat A2 = A * rotation;
	auto q2 = arma::dot(A2.col(0), theta);
	auto sx2 = arma::dot(A2.col(1), theta) * 1. + prod; // macht relativ wenig sinn, mal mit paper abgleichen
	auto croots2 = math_utils::get_complex_roots(1, A2, b, r);
	auto c2 = std::get<0>(croots2);
	auto c2_0 = std::get<1>(croots2);

	auto y = 0.;//config.y_resolution * (double)i;// steps[i];
	auto u = y + config.y_resolution * 0.5;
	auto [c, c_0] = math_utils::get_complex_roots(u, A, b, r);
	auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
	auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

	auto s = arma::dot(A.col(0), theta) * u + prod;

	auto is_spec = math_utils::is_singularity_in_layer(config.tolerance, spec_point, 0, 1 - u);
	auto is_sing = math_utils::is_singularity_in_layer(config.tolerance, sing_point, 0, 1 - u);

	integrator::gsl_integrator gslintegrator;
	integrator::gsl_integrator_2d gsl_integrator_2d;
	integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);


	//auto integration_y = integral2d.integrate_lambda(A1, b, r, 0., q1, sx1, c1, c1_0, y, y + config.y_resolution);
	//auto integration_1_minus_y = integral2d.integrate_lambda(A2, b, r, 1., q2, sx2, c2, c2_0, y, y + config.y_resolution);

	//std::cout << "intY " << integration_y << "\n";
	//std::cout << "intY-1 " << integration_1_minus_y << "\n";




	auto result = integral2d(A, b, r, theta);
	std::cout << "reuslt: " << result << std::endl;
}


auto integral_test_2d_multiple() {
	arma::mat  A{ {0, 0}, {2, 0}, {0, 2} };

	arma::vec b{ 0,-0.5,0 };

	arma::vec r{ 1, 0, 0 };

	arma::vec mu{ 1,0, 0 };

	std::vector<double> alphas;
	for (auto i = -0.5; i < 0.5; i += 0.05) {
		alphas.push_back(i);
	}
	std::vector<double> rad_list;
	for (auto i = 0.2; i < 3.8; i += 0.4) {
		rad_list.push_back(i);
	}

	config::configuration_2d config;
	config.wavenumber_k = 500;
	config.tolerance = 0.1;
	config.y_resolution = 0.0001;
	config.gauss_laguerre_nodes = 1000;

	integrator::gsl_integrator gslintegrator;
	integrator::gsl_integrator_2d gsl_integrator_2d;
	integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);

	/*alpha = 2 * pi * alpha_list(a);
	rad = rad_list(rr);

	rot_mat = [cos(alpha), -sin(alpha), 0; sin(alpha), cos(alpha), 0; 0, 0, 1];
	r = rad * rot_mat * r_0; % +[0; 0.5; 0];*/

	std::vector<std::complex<double>> results(alphas.capacity() * rad_list.capacity());
	std::vector<double> timings(alphas.capacity() * rad_list.capacity());

	std::vector<arma::vec3> r_values(alphas.capacity() * rad_list.capacity());
	std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
	auto i = 0;
	for (auto& rad : rad_list) {
		for (auto& alpha : alphas) {
			arma::mat rotation = { { cos(alpha), -sin(alpha), 0}, {sin(alpha), cos(alpha), 0}, {0, 0, 1 } };
			arma::vec3 tmp_r = (rad * rotation * r);
			r_values.push_back(tmp_r);
			i++;
		}
	}
	std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
	std::cout << "number of r_values: " << i << std::endl;
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count() << "[ms]" << std::endl; 
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	for (auto& tmp_r : r_values) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		auto result = integral2d(A, b, tmp_r, mu);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		results.push_back(result);
		timings.push_back(time);
	}

	auto average_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
	std::cout << "Time difference (average) = " << average_time << "[ms]" << std::endl;
}

int main()
{
	/* 
	// use this as tesT?=
	arma::mat  A{{0, 0}, {1, 0}, {1, 1}};

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };
	arma::vec mu{ 1,-0.5, -0.5 };
	auto y = 0.;
	auto k = 100;
	auto left_split = 0.;
	auto right_split = 1.;
	//integrator::integrator
	//const std::complex<double>(const integrand_fun fun, const std::complex<double> first_split_point, const std::complex<double> second_split_point)
	auto integrator = [](const integrator::integrand_fun fun, const std::complex<double> first_split_point, const std::complex<double> second_split_point) -> std::complex<double> {
		return  -0.090716 + 0.259133i;
	};
	integral::integral_1d integral1d(k, integrator , 0.1);


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
	//integral_test_1d();


	/*
	std::cout << "Resolution 0.1" << std::endl;
	for (auto i = 0; i < 5; i++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		integral_test_2d(0.1);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	}

	std::cout << "Resolution 0.01" << std::endl;
	for (auto i = 0; i < 5; i++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		integral_test_2d(0.01);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	}

	std::cout << "Resolution 0.001" << std::endl;
	for (auto i = 0; i < 5; i++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		integral_test_2d(0.001);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	}

	std::cout << "Resolution 0.0001" << std::endl;
	for (auto i = 0; i < 5; i++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		integral_test_2d(0.0001);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	}*/

	/*eval_2d_article(100);
	eval_2d_article(500);
	eval_2d_article(1000);
	eval_2d_article(3000);
	eval_2d_article(5000);*/

	check_accuracy();
	
	//integral_test_2d_multiple();
	//integral_test_2d(1);
	return 0;
}

