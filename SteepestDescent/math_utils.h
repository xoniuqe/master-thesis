#pragma once
#include <vector>
#include <algorithm>
#include <numeric>

#include <complex>
#include <armadillo>
#include "datatypes.h"
#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif
#include <math.h>
#include <type_traits>

namespace math_utils {
	using namespace std::complex_literals;
	
	template<class T> struct is_complex : std::false_type {};
	template<class T> struct is_complex<std::complex<T>> : std::true_type {};



	

	

	auto get_laguerre_points(const int n) {

	}

	auto lagpts(int n) {
		/*alpha = 2 * (1:n) - 1;  beta = 1:(n - 1); % 3 - term recurrence coeffs
			T = diag(beta, 1) + diag(alpha) + diag(beta, -1);% Jacobi matrix
			[V, D] = eig(T);% eigenvalue decomposition
			[x, indx] = sort(diag(D));% Laguerre points
			w = V(1, indx). ^ 2;% Quadrature weights
			v = sqrt(x).*abs(V(1, indx)).';        % Barycentric weights
			v = v. / max(v); v(2:2 : n) = -v(2:2 : n);*/
		std::vector<double> alpha(n);
		std::iota(std::begin(alpha), std::end(alpha), 1);
		std::for_each(std::begin(alpha), std::end(alpha), [](auto& x) { x = 2. * x - 1.; });
		std::vector<double> beta(n-1);
		
		std::iota(std::begin(beta), std::end(beta), 0);
		arma::mat T = arma::diagmat(arma::vec(beta), 1) + arma::diagmat(arma::vec(alpha)) + arma::diagmat(arma::vec(beta), -1);
		arma::cx_vec eigval;
		arma::cx_mat eigvec;
		auto eigen = arma::eig_gen(eigval, eigvec, T);
		//auto diag = arma::diagvec(eigvec);

	/// <summary>
	/// singularity := definitionsl�cke
	/// </summary>
	/// <param name="complex_root"></param>
	/// <param name="q"></param>
	/// <param name="k"></param>
	/// <param name="s"></param>
	/// <returns></returns>
	auto calculate_singularities_ODE(const datatypes::complex_root complex_root, const double q, const double k, const double s) 
	{  
		auto C = complex_root.c_0 - q * q;
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);
		auto sqc = std::sqrt(complex_root.c_0);
		auto cTimesCconj = rc * rc + ic * ic;
		auto tmp = (rc * rc * q * q + cTimesCconj * C);
		auto tmp2 = complex_root.c_0 * rc * rc;
		auto tmp3 = std::sqrt(tmp - tmp2);
		auto tmp4 = rc * q + s;
		auto n_sing = (M_PI / k) * (floor((k / M_PI) * (tmp3 + tmp4)) + 1) - s;
		auto e_sing = ((complex_root.c_0 * rc) - (q * n_sing)) / C;
		auto r_sing = std::sqrt((n_sing * n_sing - complex_root.c_0 * cTimesCconj) / C + e_sing * e_sing);
		return std::make_tuple(n_sing, e_sing, r_sing);
	}
	constexpr auto decide_split_points(const double left_split_zero, const double right_split_zero, const double left_split_point, const double right_split_point)
	{
		double first_split_point = 0.;
		double second_split_point = 0.;

		if (left_split_zero <= left_split_point) {
			first_split_point = left_split_point;
		}
		if (left_split_zero >= right_split_point) {
			first_split_point = right_split_point;
		}
		if (left_split_point < left_split_zero && left_split_zero < right_split_point) {
			first_split_point = left_split_zero;
		}

		if (right_split_zero >= right_split_point) {
			second_split_point = right_split_point;
		}
		if (right_split_zero <= left_split_point) {
			second_split_point = left_split_point;
		}

		if (left_split_point < right_split_zero && right_split_zero < right_split_point) {
			second_split_point = right_split_zero;
		}

		return std::make_tuple(first_split_point, second_split_point);
	}

	auto get_split_points_sing(const double q, const double k, const double s, const datatypes::complex_root complex_root, const double left_split_point, const double right_split_point)
	{
		auto [n_sing, e_sing, r_sing] = calculate_singularities_ODE(complex_root, q, k, s);

		auto left_split_zero = e_sing - r_sing;
		auto right_split_zero = e_sing + r_sing;


		return decide_split_points(left_split_zero, right_split_zero, left_split_point, right_split_point);
	}



	auto get_split_points_spec(const double q, const double k, const double s, const datatypes::complex_root complex_root, const double left_split_point, const double right_split_point)
	{
		auto C = complex_root.c_0 - q * q;
		auto rc = std::real(complex_root.c);		
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;


		auto n0_spec = M_PI / k * (std::floor(k / M_PI * (rc * q * q / q + s))) - s;
		auto n1_spec = M_PI / k * (std::floor(k / M_PI * (rc * q * q / q + s)) + 1) - s;
		auto e0_spec = (complex_root.c_0 * rc - q * n0_spec) / C;
		auto e1_spec = (complex_root.c_0 * rc - q * n1_spec) / C;

		double left_zero_split, right_zero_split;
		double first_split_point, second_split_point;
		if (q < 0) {
			left_zero_split = e1_spec + std::sqrt((n1_spec * n1_spec - complex_root.c_0 * cTimesCconj) / C + e1_spec * e1_spec);
			right_zero_split = e0_spec + std::sqrt((n0_spec * n0_spec - complex_root.c_0 * cTimesCconj) / C + e0_spec * e0_spec);
		}
		else {
			left_zero_split = e0_spec - std::sqrt((n0_spec * n0_spec - complex_root.c_0 * cTimesCconj) / C + e0_spec * e0_spec);
			right_zero_split = e1_spec - std::sqrt((n1_spec * n1_spec - complex_root.c_0 * cTimesCconj) / C + e1_spec * e1_spec);
		}

		return decide_split_points(left_zero_split, right_zero_split, left_split_point, right_split_point);
	}
	
	auto get_singularity_for_ODE(const double q, const datatypes::complex_root complex_root) -> std::complex<double> {
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		auto F = std::sqrt(std::complex<double>(rc * rc - (q * q / complex_root.c_0 * cTimesCconj - rc * rc) / (q * q / complex_root.c_0 - 1)));

		return q < 0 ? rc + F :rc - F;
	}

	auto get_spec_point(const double q, const datatypes::complex_root complex_root) {
		auto C = std::real(complex_root.c);
		auto c_real_squared = std::pow(C, 2);
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		auto K = std::sqrt(1. / (complex_root.c_0 - q * q) * (q * q * C * C - complex_root.c_0 * cTimesCconj) + C * C);

		return q < 0 ? C + K : C - K;
	}

	auto get_complex_roots(const double y, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r) -> auto {
		auto A_1 = A.col(0);
		auto A_2 = A.col(1);
		auto A_2copy = arma::vec(A_2);
		A_2copy *= y;
		A_2copy += b - r;

		auto real_c = -arma::dot(A_1, A_2copy) / arma::dot(A_1, A_1);

		auto P_rc = calculate_P_x(real_c, y, A, b, r);

		double c_0;
		if (abs(real_c) > 0.000000001) {
			auto x = calculate_P_x(0, y, A, b, r);
			c_0 = (x - P_rc) / (real_c * real_c);
		}
		else {
			auto x = partial_derivative_P_x(1, y, A, b, r);
			c_0 = 1.0 / 2.0 * x;

		}

		auto c = real_c + std::sqrt((P_rc / c_0)) * 1i;
		return std::make_tuple(c, c_0);
	}
}
