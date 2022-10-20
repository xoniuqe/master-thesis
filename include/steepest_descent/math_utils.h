#pragma once
#ifndef ST_MATH_UTILS
#define ST_MATH_UTILS
#include <complex>
#include "datatypes.h"
#include "complex_comparison.h"
#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif
#include <math.h>
#include <type_traits>

namespace math_utils {
	using namespace std::complex_literals;

	using namespace complex_comparison;

	typedef std::function<std::complex<double>(double)> green_fun;
	typedef std::function<green_fun(const double& k, const std::complex<double>& y, const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double& q, const std::complex<double>& s)> green_fun_generator;

	inline auto floor(std::complex<double> && value) -> std::complex<double> {
		return  std::floor(std::real(value)) + std::floor(std::imag(value)) * 1.i;
	}

	template<typename T>
	inline auto get_q(const arma::subview_col<T>& A, const arma::vec3& theta) -> double {
		return arma::dot(A, theta);
	}


	template<typename T>
	constexpr auto get_singularity_for_ODE(const T q, const datatypes::complex_root complex_root) -> std::complex<double> {
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		// streng genommen nur real!
		auto F = std::sqrt(std::complex<double>(rc * rc - (q * q / complex_root.c_0 * cTimesCconj - rc * rc) / (q * q / complex_root.c_0 - 1.)));

		return std::real(q) < 0. ? rc + F : rc - F;
	}

	/// <summary>
	/// singularity := definitionsl�cke
	/// </summary>
	/// <param name="complex_root"></param>
	/// <param name="q"></param>
	/// <param name="k"></param>
	/// <param name="s"></param>
	/// <returns></returns>
	template<typename T>
	constexpr auto calculate_singularities_ODE(const datatypes::complex_root complex_root, const double q, const double k, const T s) -> std::tuple<std::complex<double>, std::complex<double>, std::complex<double>>
	{
		auto C = complex_root.c_0 - q * q;
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);
		auto cTimesCconj = rc * rc + ic * ic;
		auto tmp = (rc * rc * q * q + cTimesCconj * C);
		auto tmp2 = complex_root.c_0 * rc * rc;
		auto tmp3 = std::sqrt(tmp - tmp2);
		auto tmp4 = rc * q + s;
		auto n_sing = (M_PI / k) * (floor((k / M_PI) * (tmp3 + tmp4)) + 1.) - s;
		auto e_sing = ((complex_root.c_0 * rc) - (q * n_sing)) / C;
		auto r_sing = std::sqrt((n_sing * n_sing - complex_root.c_0 * cTimesCconj) / C + e_sing * e_sing);
		return std::make_tuple(n_sing, e_sing, r_sing);
	}

	constexpr auto decide_split_points(const std::complex<double> left_split_zero, const std::complex<double> right_split_zero, const std::complex<double> left_split_point, const std::complex<double> right_split_point) -> std::tuple<std::complex<double>, std::complex<double>>
	{
		std::complex<double> first_split_point;
		std::complex<double> second_split_point;

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



	template<typename T>
	constexpr auto get_split_points_sing(const double q, const double k, const T s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point) -> std::tuple<std::complex<double>, std::complex<double>>
	{
		auto [n_sing, e_sing, r_sing] = calculate_singularities_ODE(complex_root, q, k, s);

		auto left_split_zero = e_sing - r_sing;
		auto right_split_zero = e_sing + r_sing;

		return decide_split_points(left_split_zero, right_split_zero, left_split_point, right_split_point);
	}

	template<typename T>
	constexpr auto get_split_points_spec(const double q, const double k, const T s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point) -> std::tuple<std::complex<double>, std::complex<double>>
	{
		auto C = complex_root.c_0 - q * q;
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;


		auto n0_spec = M_PI / k * (floor(k / M_PI * (rc * q * q / q + s))) - s;
		auto n1_spec = M_PI / k * (floor(k / M_PI * (rc * q * q / q + s)) + 1.) - s;
		auto e0_spec = (complex_root.c_0 * rc - q * n0_spec) / C;
		auto e1_spec = (complex_root.c_0 * rc - q * n1_spec) / C;
		//streng genommen auch nur real
		std::complex<double> left_zero_split, right_zero_split;
		std::complex<double> first_split_point, second_split_point;
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

	


	auto get_spec_point(const double q, const datatypes::complex_root complex_root)->std::complex<double>;


	/// <summary>
	/// Takes only positive real values
	/// </summary>
	/// <param name="x"></param>
	/// <param name="y"></param>
	/// <param name="A"></param>
	/// <param name="b"></param>
	/// <param name="r"></param>
	/// <returns></returns>
	template<typename T1, typename T2>
	auto calculate_P_x(const T1 x, const T2 y, const arma::mat& A, const arma::vec3& b, const arma::vec3& r)->auto {
		auto calculate_row = [&](const size_t i) {
			auto first_val = A.at(i, 0);
			auto second_val = A.at(i, 1);
			auto b_element = b.at(i);
			auto r_element = r.at(i);
			auto tmp = ((first_val * x + second_val * y + b_element - r_element));
			return tmp * tmp;
		};
		return calculate_row(0) + calculate_row(1) + calculate_row(2);
	}
	/// <summary>
	/// Calculates the partial derivative in x direction.
	/// </summary>
	/// <param name="x">Barycentrical parameter x of the triangle </param>
	/// <param name="y">Barycentrical parameter y of the triangle </param>
	/// <param name="A">Jacobian matrix of the triangle </param>
	/// <param name="b">Affine transformation Ax + b </param>
	/// <param name="r">View vector </param>
	/// <returns>The partial derivative in x direction </returns>
	template<typename Tnumeric>
	auto partial_derivative_P_x(const double x, const Tnumeric y, const arma::mat& A, const arma::vec3& b, const arma::vec3& r)->auto
	{
		auto calculate_row = [&](const size_t i) {
			auto first_val = A.at(i, 0);
			auto second_val = A.at(i, 1);
			auto b_element = b.at(i);
			auto r_element = r.at(i);
			return first_val * 2 * (first_val * x + second_val * y + b_element - r_element);
		};
		return calculate_row(0) + calculate_row(1) + calculate_row(2);
	}

	auto get_complex_roots(const std::complex<double> y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->std::tuple<std::complex<double>, double>;

	auto get_complex_roots(const double y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r) -> std::tuple<std::complex<double>, double>;


	constexpr auto is_singularity_in_layer(const double tolerance, const std::complex<double> sing_point, const double first_split, const double second_split) -> bool {
		return (std::real(sing_point) >= (first_split - tolerance) && std::real(sing_point) <= (second_split + tolerance) && std::abs(std::imag(sing_point)) <= std::numeric_limits<double>::epsilon());
	}}
#endif