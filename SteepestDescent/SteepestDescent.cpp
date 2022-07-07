// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include <complex> 
#include <variant>
#include <tuple>

#include <armadillo>
#include <steepest_descent/gauss_laguerre.h>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/datatypes.h>
#include <steepest_descent/path_utils.h>
#include <steepest_descent/integrator.h>
#include <steepest_descent/integral.h>

#include <type_traits>
using namespace std::complex_literals;
typedef std::variant<std::tuple<std::complex<double>, std::complex<double>>, std::complex<double>> splitting_point;

/*
/// <summary>
/// 
/// </summary>
/// <param name="x"></param>
/// <param name="A"></param>
/// <param name="b"></param>
/// <param name="r"></param>
/// <returns></returns>
auto calculate_P_x(const datatypes::vector& x, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r) {
	arma::vec tmp = b - r;

	tmp += A * x;
	auto result = DOT_PRODUCT(tmp, tmp);
	
	return result;
}
*/

constexpr auto calculate_laguerre_point(const int k, const int a, const double x) {
	if (k == 0) {
		return 1.;
	}
	if (k == 1) {
		return 1. + a - x;
	}
	auto l_k_prev = calculate_laguerre_point(k - 1, a, x);
	auto l_k_prev_prev = calculate_laguerre_point(k - 2, a, x);
	auto n_k = (double)k - 1.;
	auto left_factor = 2. * n_k + 1. + a - x;
	auto left = left_factor * l_k_prev;

	auto right_factor = n_k + a;
	auto right = right_factor * l_k_prev_prev;
	auto k_plus_one_inv = 1. / (n_k + 1.);
	return (left - right) * k_plus_one_inv;
}



auto calculate_laguerre_points_and_weights(int n) {
	std::vector<double> alpha(n);
	std::iota(std::begin(alpha), std::end(alpha), 1);
	std::for_each(std::begin(alpha), std::end(alpha), [](auto& x) { x = 2. * x - 1.;  });
	std::vector<double> beta(n);
	std::iota(std::begin(beta), std::end(beta), 1);


	arma::mat T(n, n);
	for (auto i = 0; i < n; i++) {
		T.at(i, i) = alpha[i];
		if (i + 1 < n) {
			T.at(i, i + 1) = beta[i];
			T.at(i + 1, i) = beta[i];
		}
	}

	arma::mat evec;
	arma::vec laguerre_points;
	auto result = arma::eig_sym(laguerre_points, evec , T);
	std::cout << "result: " << result << std::endl;
	auto diag = evec.diag();

	arma::vec quadrature_weights;
	for (auto i = 0; i < n; i++) {
		quadrature_weights[i] = evec.at(0, i) * evec.at(0, i);
	}


	arma::vec barycentric_weights;
	double max_value = -1.;
	for (auto i = 0; i < n; i++) {
		barycentric_weights[i] = std::abs(evec.at(0, i)) * std::sqrt(laguerre_points[i]);
		if (barycentric_weights[i] > max_value)
			max_value = barycentric_weights[i];
	}

	barycentric_weights /= max_value;
	barycentric_weights *= -1.;
		

	return std::make_tuple(laguerre_points, quadrature_weights, barycentric_weights);
}


auto calculate_splitting_points(const std::complex<double> c, const double c_0, const double q) -> std::tuple<splitting_point, splitting_point> {
	if (arma::arma_isinf(q) != 0) {
		// c_s is bi-valued 
		auto c_s = std::make_tuple(c, std::conj(c));
		auto c_r = std::real(c) + 0i;
		return std::make_tuple(c_s, c_r);
		//auto c_s = std::make_tuple(c, gsl_complex_conjugate(gsl_complex(c)));

	}

	if (std::abs(q) < std::numeric_limits<double>::min()) {
		auto c_s = std::real(c) + 0i;
		auto c_r = std::make_tuple(c, std::conj(c));
		return std::make_tuple(c_s, c_r);
	}

	if (std::abs(q) - sqrt(c_0) < std::numeric_limits<double>::min()) {
		//return std::make_tuple(arma::max_, GSL_POSINF);
		//Todo: 
	}
	auto c_real_cube = std::real(c) * std::real(c);
	auto c_cube_abs = std::abs(c) * std::abs(c);
	auto q_cubed = q * q;
	auto K_c_s = sqrt(c_real_cube + (c_0 * c_real_cube - c_cube_abs * q_cubed) / (c_0 - q_cubed));
	auto K_c_r = sqrt(c_real_cube + (q_cubed * c_real_cube - c_0 * c_cube_abs) / (c_0 - q_cubed));
	// here there seems to be no difference between the cases |q| < sqrt(c_0) and |q| > sqrt(c_0)
	//if (std::abs(q) < sqrt(c_0)) {
	if (q < 0) {
		auto c_s = std::real(c) + K_c_s;
		auto c_r = std::real(c) + K_c_r;
		return std::make_tuple(c_s, c_r);
	}
	auto c_s = std::real(c) - K_c_s;
	auto c_r = std::real(c) - K_c_r;
	return std::make_tuple(c_s, c_r);
	/* }
	else {

	}*/
}




auto integrate_1d(const double y, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r, const datatypes::vector& mu, const double k, const double left_split_point, const double right_split_point, std::tuple<std::vector<double>, std::vector<double>> laguerre_points)
{
	auto a_1 = A.col(0);
	auto a_2 = A.col(1);
	auto q = arma::dot(a_1, mu);
	auto s = arma::dot(a_2, mu) * y + arma::dot(mu, b);

	auto [c, c_0] = math_utils::get_complex_roots(y, A, b, r);
	auto sing_point = math_utils::get_singularity_for_ODE(q, datatypes::complex_root{ c, c_0 });
	auto spec_poitn = math_utils::get_spec_point(q, datatypes::complex_root{ c, c_0 });

	if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) 
	{
		sing_point = std::real(sing_point);
	}
}

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


auto integral_test()
{
	arma::mat  A{ {0, 0}, {1, 0}, {1, 1 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };

	arma::vec mu{ 1,-0.5, -0.5 };
	auto y = 0.;
	auto k = 100;
	auto left_split = 0.; 
	auto right_split = 1.;

	std::cout << "Px: " << math_utils::calculate_P_x(left_split, y, A, b, r) << std::endl;
	std::cout << "Px: " << math_utils::calculate_P_x(right_split, y, A, b, r) << std::endl;

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
	auto tol = 0.1;
	if (std::real(spec_point) >= (left_split - tol) && std::real(spec_point) <= (right_split + tol) && std::abs(std::imag(spec_point)) <= std::numeric_limits<double>::epsilon()) {
		std::cout << "spec point" << spec_point <<  std::endl;
	}

	if (std::real(sing_point) >= (left_split - tol) && std::real(sing_point) <= (right_split + tol) && std::abs(std::imag(sing_point)) <= std::numeric_limits<double>::epsilon()) {
		std::cout << "sing point" << sing_point << std::endl;
		auto [sp1, sp2] = math_utils::get_split_points_sing(q, k, s, { c, c_0 }, left_split, right_split);

		std::cout << "left: " << left_split << std::endl;
		std::cout << "right: " << right_split << std::endl;

		std::cout << "sp1: " << sp1 << std::endl;
		std::cout << "sp2: " << sp2 << std::endl;

		auto path1 = path_utils::get_weighted_path(left_split, y, A, b, r, q, k, s, { c, c_0 }, sing_point);
		auto path2 = path_utils::get_weighted_path(sp1, y, A, b, r, q, k, s, { c, c_0 }, sing_point);

		auto I1 = gauss_laguerre::calculate_integral_cauchy(path1, path2, nodes, weights);

	
		auto path3 = path_utils::get_weighted_path(sp2, y, A, b, r, q, k, s, { c, c_0 }, sing_point);
		std::cout << "weighted path: " << path3(1) << std::endl;
		auto path4 = path_utils::get_weighted_path(right_split, y, A, b, r, q, k, s, { c, c_0 }, sing_point);
		std::cout << "weighted path: " << path4(1) << std::endl;

	
		auto I2 = gauss_laguerre::calculate_integral_cauchy(path3, path4, nodes, weights);


		std::cout << "I1: " << I1 << std::endl;
		std::cout << "I2: " << I2 << std::endl;

		//just the "normal" integrate is missing
		//    greenFun1D = @(x) exp(1i*k*(Px(x,y,A,b,r).^(1/2)+q*x+s)).*Px(x,y,A,b,r).^(-1/2); 
		//matlab:  integral(greenFun1D,splitPt1,splitPt2, "ArrayValued", "True") 
		auto x = -0.090716 + 0.259133i;
		std::cout << "result: " << I1 + x + I2 << std::endl;
		return;
	}

	std::cout << "no singularity: exit; did not implement 'integral' yet" << std::endl;
	return;



		/*   
    if specPoint >= lSp-tol && specPoint <= rSp+tol && imag(specPoint) == 0    %specPoint lies in the integration segment.
        string = 'Spec';
    end


    if singPoint >= lSp-tol && singPoint <= rSp+tol && imag(singPoint) == 0    %singPoint lies in the integration segment.
        string = 'Sing';
    end*/
}




auto integral_test2()
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
	integral::integral_1d integral1d(k, gslintegrator, 0.1);
	auto res = integral1d(A, b, r, mu, y, left_split, right_split);
	std::cout << "result2: " << res << std::endl;
	return;



	/*
if specPoint >= lSp-tol && specPoint <= rSp+tol && imag(specPoint) == 0    %specPoint lies in the integration segment.
	string = 'Spec';
end


if singPoint >= lSp-tol && singPoint <= rSp+tol && imag(singPoint) == 0    %singPoint lies in the integration segment.
	string = 'Sing';
end*/
}



int main()
{
	arma::mat  A{ {1, 1}, {1, 1,} , {0 ,0 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 0, 1, 2 };

	arma::vec mu{ 1,4,0 };

	auto k = 10;
	auto s = 3;

	integral_test();
	integral_test2();
	/*
	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

	auto q = -std::sqrt(c_0);

	auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(10);
	auto [path, derivative] = 	path_utils::get_complex_path(0, 0, A, b, r, q, {c, c_0}, 0);
	//gauss_laguerre::calculate_integral()
	//setup_1d_test();
	test_split();*/

	return 0;
}

