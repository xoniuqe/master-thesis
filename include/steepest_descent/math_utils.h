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
	auto calculate_P_x(const double x, const double y, const datatypes::matrix& A, const  arma::vec3& b, const arma::vec3& r) -> double;

	/*template<typename T>
	auto calculate_P_x(const T x, const double y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r) -> auto {
		//arma::vec X{ x, y };
		arma::vec result = A * { x , y } + b - r;
		return result.at(0) + result.at(1) + result.at(2);
	}*/
	auto calculate_P_x(const std::complex<double> x, const double y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->std::complex<double>;

	auto calculate_P_x(const std::complex<double> x, const std::complex<double> y, const datatypes::matrix& A, const arma::vec3& b, const  arma::vec3& r)->std::complex<double>;

	/// <summary>
	/// Calculates the partial derivative in x direction.
	/// </summary>
	/// <param name="x">Barycentrical parameter x of the triangle </param>
	/// <param name="y">Barycentrical parameter y of the triangle </param>
	/// <param name="A">Jacobian matrix of the triangle </param>
	/// <param name="b">Affine transformation Ax + b </param>
	/// <param name="r">View vector </param>
	/// <returns>The partial derivative in x direction </returns>
	auto partial_derivative_P_x(const double x, const double y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->double;
	auto partial_derivative_P_x(const double x, const std::complex<double> y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->std::complex<double>;


	auto get_complex_roots(const std::complex<double> y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r)->std::tuple<std::complex<double>, double>;

	auto get_complex_roots(const double y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r) -> std::tuple<std::complex<double>, double>;

	auto singularity_as_real(const std::complex<double> singularity, const std::complex<double> c) -> bool;

	auto layer_contains_singularity(const std::complex<double> candiate, const double point1, const double point2) -> bool;
}
