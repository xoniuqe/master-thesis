#pragma once

#include <complex>
#include "datatypes.h"
#include "complex_comparison.h"
#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif
#include <type_traits>

namespace math_utils {
	using namespace std::complex_literals;

	using namespace complex_comparison;

	
	template<class T> struct is_complex : std::false_type {};
	template<class T> struct is_complex<std::complex<T>> : std::true_type {};

	template<class T>
	auto floor(T value) -> T {
		return std::floor(value);
	}

	template<class T, std::is_base_of<std::true_type, is_complex<T>> = true>
	auto floor(T value) -> T {
		return  floor(std::real(value)) + floor(std::imag(value)) * 1i;
	}


	/// <summary>
	/// singularity := definitionsl�cke
	/// </summary>
	/// <param name="complex_root"></param>
	/// <param name="q"></param>
	/// <param name="k"></param>
	/// <param name="s"></param>
	/// <returns></returns>
	auto calculate_singularities_ODE(const datatypes::complex_root complex_root, const double q, const double k, const double s)->std::tuple<std::complex<double>, std::complex<double>, std::complex<double>>;
	constexpr auto decide_split_points(const std::complex<double> left_split_zero, const std::complex<double> right_split_zero, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;
	auto get_split_points_sing(const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;



	auto get_split_points_spec(const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point)->std::tuple<std::complex<double>, std::complex<double>>;
	
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
	auto calculate_P_x(const double x, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r) -> double; 


	auto calculate_P_x(const std::complex<double> x, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r)->std::complex<double>;


	/// <summary>
	/// Calculates the partial derivative in x direction.
	/// </summary>
	/// <param name="x">Barycentrical parameter x of the triangle </param>
	/// <param name="y">Barycentrical parameter y of the triangle </param>
	/// <param name="A">Jacobian matrix of the triangle </param>
	/// <param name="b">Affine transformation Ax + b </param>
	/// <param name="r">View vector </param>
	/// <returns>The partial derivative in x direction </returns>
	auto partial_derivative_P_x(const double x, const double y, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r)->double;


	auto get_complex_roots(const double y, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r) -> std::tuple<std::complex<double>, double>;
}
