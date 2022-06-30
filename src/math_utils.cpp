#include <steepest_descent/math_utils.h>
#include <steepest_descent/datatypes.h>

#include <vector>
#include <algorithm>
#include <numeric>

#include <complex>
#include <armadillo>
#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif
#include <math.h>
#include <type_traits>

namespace math_utils {
	using namespace std::complex_literals;


	/// <summary>
	/// singularity := definitionsl�cke
	/// </summary>
	/// <param name="complex_root"></param>
	/// <param name="q"></param>
	/// <param name="k"></param>
	/// <param name="s"></param>
	/// <returns></returns>
	auto calculate_singularities_ODE(const datatypes::complex_root complex_root, const double q, const double k, const double s) -> std::tuple<std::complex<double>, std::complex<double>, std::complex<double>>
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

	auto get_split_points_sing(const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point) -> std::tuple<std::complex<double>, std::complex<double>>
	{
		auto [n_sing, e_sing, r_sing] = calculate_singularities_ODE(complex_root, q, k, s);

		auto left_split_zero = e_sing - r_sing;
		auto right_split_zero = e_sing + r_sing;


		return decide_split_points(left_split_zero, right_split_zero, left_split_point, right_split_point);
	}



	auto get_split_points_spec(const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> left_split_point, const std::complex<double> right_split_point) -> std::tuple<std::complex<double>, std::complex<double>>
	{
		auto C = complex_root.c_0 - q * q;
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;


		auto n0_spec = M_PI / k * (std::floor(k / M_PI * (rc * q * q / q + s))) - s;
		auto n1_spec = M_PI / k * (std::floor(k / M_PI * (rc * q * q / q + s)) + 1) - s;
		auto e0_spec = (complex_root.c_0 * rc - q * n0_spec) / C;
		auto e1_spec = (complex_root.c_0 * rc - q * n1_spec) / C;

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

	auto is_within_split_point_tolerances(const std::complex<double> point, const double left_split, const double right_split, const double tolerance) -> bool {
		return std::real(point) >= (left_split - tolerance) && std::real(point) <= (right_split + tolerance) && std::abs(std::imag(point)) <= std::numeric_limits<double>::epsilon();
	}

	auto get_singularity_for_ODE(const double q, const datatypes::complex_root complex_root) -> std::complex<double> {
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		auto F = std::sqrt(std::complex<double>(rc * rc - (q * q / complex_root.c_0 * cTimesCconj - rc * rc) / (q * q / complex_root.c_0 - 1)));

		return q < 0 ? rc + F : rc - F;
	}

	auto get_spec_point(const double q, const datatypes::complex_root complex_root) -> std::complex<double> {
		auto C = std::real(complex_root.c);
		auto c_real_squared = std::pow(C, 2);
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		auto K = std::sqrt(std::complex<double>(1. / (complex_root.c_0 - q * q) * (q * q * C * C - complex_root.c_0 * cTimesCconj) + C * C));

		return q < 0 ? C + K : C - K;
	}

	/// <summary>
	/// Takes only positive real values
	/// </summary>
	/// <param name="x"></param>
	/// <param name="y"></param>
	/// <param name="A"></param>
	/// <param name="b"></param>
	/// <param name="r"></param>
	/// <returns></returns>
	auto calculate_P_x(const double x, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r) -> double {
		//fx = ((A(1, 1) * x + A(1, 2) * y + b(1) - r(1)). ^ 2 + (A(2, 1) * x + A(2, 2) * y + b(2) - r(2)). ^ 2 + (A(3, 1) * x + A(3, 2) * y + b(3) - r(3)). ^ 2);
		auto calculate_row = [&](const int i) {
			auto first_val = MATRIX_GET(A, i, 0);
			auto second_val = MATRIX_GET(A, i, 1);
			auto b_element = VECTOR_GET(b, i);
			auto r_element = VECTOR_GET(r, i);
			auto tmp = ((first_val * x + second_val * y + b_element - r_element));
			return tmp * tmp;
		};
		return calculate_row(0) + calculate_row(1) + calculate_row(2);
	}

	auto calculate_P_x(const std::complex<double> x, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r) -> std::complex<double> {
		//fx = ((A(1, 1) * x + A(1, 2) * y + b(1) - r(1)). ^ 2 + (A(2, 1) * x + A(2, 2) * y + b(2) - r(2)). ^ 2 + (A(3, 1) * x + A(3, 2) * y + b(3) - r(3)). ^ 2);
		auto calculate_row = [&](const int i) {
			auto first_val = MATRIX_GET(A, i, 0);
			auto second_val = MATRIX_GET(A, i, 1);
			auto b_element = VECTOR_GET(b, i);
			auto r_element = VECTOR_GET(r, i);
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
	auto partial_derivative_P_x(const double x, const double y, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r) -> double {
		auto calculate_row = [&](const int i) {
			auto first_val = MATRIX_GET(A, i, 0);
			auto second_val = MATRIX_GET(A, i, 1);
			auto b_element = VECTOR_GET(b, i);
			auto r_element = VECTOR_GET(r, i);
			return first_val * 2 * (first_val * x + second_val * y + b_element - r_element);
		};
		return calculate_row(0) + calculate_row(1) + calculate_row(2);
	}


	auto get_complex_roots(const double y, const datatypes::matrix& A, const datatypes::vector& b, const datatypes::vector& r) -> std::tuple<std::complex<double>, double> {
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
