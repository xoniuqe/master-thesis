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


	auto get_spec_point(const double q, const datatypes::complex_root complex_root) -> std::complex<double> {
		auto C = std::real(complex_root.c);
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		auto K = std::sqrt(std::complex<double>(1. / (complex_root.c_0 - q * q) * (q * q * C * C - complex_root.c_0 * cTimesCconj) + C * C));

		return q < 0 ? C + K : C - K;
	}

	auto get_complex_roots(const std::complex<double> y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r) -> std::tuple<std::complex<double>, double> {
		auto A_1 = A.col(0);
		auto A_2 = A.col(1);

		
		auto A2_imaginary = arma::vec3();
		A2_imaginary.zeros();

		auto b_zeros = arma::vec3();
		b_zeros.zeros();

		auto r_zeros = arma::vec3();
		r_zeros.zeros();
		auto b_complex = arma::cx_vec3(b, b_zeros);
		auto r_complex = arma::cx_vec3(r, r_zeros);

		auto A_2copy = arma::cx_vec3(A_2, A2_imaginary);
		A_2copy *= y;
		A_2copy += b_complex - r_complex;

		auto real_c = -arma::dot(A_1, A_2copy) / arma::dot(A_1, A_1);

		auto P_rc = calculate_P_x(real_c, y, A, b, r);

		double c_0;
		if (abs(real_c) > 0.000000001) {
			auto x = calculate_P_x(0, y, A, b, r);
			c_0 = std::real((x - P_rc) / (real_c * real_c));
		}
		else {
			auto x = partial_derivative_P_x(1, y, A, b, r);
			c_0 = std::real(1.0 / 2.0 * x);

		}

		auto c = real_c + std::sqrt((P_rc / c_0)) * 1i;
		return std::make_tuple(c, c_0);
	}

	auto get_complex_roots(const double y, const datatypes::matrix& A, const arma::vec3& b, const arma::vec3& r) -> std::tuple<std::complex<double>, double> {
		auto A_1 = A.col(0);
		auto A_2 = A.col(1);


		auto A_2copy = arma::vec3(A_2);
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

		auto c = real_c + std::sqrt((P_rc / c_0)) * 1.i;
		return std::make_tuple(c, c_0);
	}
}
