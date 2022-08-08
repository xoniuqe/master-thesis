#pragma once

#include <complex>
#include "datatypes.h"
#include "complex_comparison.h"
#ifdef _WIN32
//#include <corecrt_math_defines.h>
#endif
#include <type_traits>

namespace math_utils {
	using namespace std::complex_literals;

	using namespace complex_comparison;

	inline auto floor(std::complex<double> && value) -> std::complex<double> {
		return  std::floor(std::real(value)) + std::floor(std::imag(value)) * 1.i;
	}

	auto calculate_singularities_ODE(const datatypes::complex_root complex_root, const double q, const double k, const std::complex<double> s)->std::tuple<std::complex<double>, std::complex<double>, std::complex<double>>;



	/// <summary>
	/// singularity := definitionsl�cke
	/// </summary>
	/// <param name="complex_root"></param>
	/// <param name="q"></param>
	/// <param name="k"></param>
	/// <param name="s"></param>
	/// <returns></returns>
	auto calculate_singularities_ODE(const datatypes::complex_root complex_root, const double q, const double k, const double s)->std::tuple<std::complex<double>, std::complex<double>, std::complex<double>>;
	
	auto get_split_points_sing(const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;

	auto get_split_points_sing(const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;

	auto get_split_points_spec(const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;

	auto get_split_points_spec(const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;
	
	auto get_singularity_for_ODE(const std::complex<double> q, const datatypes::complex_root complex_root)->std::complex<double>;

	constexpr auto decide_split_points(const std::complex<double> left_split_zero, const std::complex<double> right_split_zero, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;

	auto get_singularity_for_ODE(const double q, const datatypes::complex_root complex_root)->std::complex<double>;

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
	auto calculate_P_x(const T1 x, const T2 y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->auto {
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
	auto partial_derivative_P_x(const double x, const Tnumeric y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->auto
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

	auto singularity_as_real(const std::complex<double> singularity, const std::complex<double> c) -> bool;

	auto is_singularity_in_layer(const double tolerance, const std::complex<double> candidate, const double first_point, const double second_point) -> bool;
}
