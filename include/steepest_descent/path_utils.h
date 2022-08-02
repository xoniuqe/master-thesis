#pragma once

#include "math_utils.h"
#include <armadillo>
#include <complex>

namespace path_utils {

	using namespace std::literals::complex_literals;
	typedef std::function <const std::complex<double>(const double t)> path_function;


	auto generate_K_x(const std::complex<double> x, const std::complex<double> P_x, const double q)->std::function<const std::complex<double>(const double t)>;

	/*
	Idea: split this function up into path calculation and derivative. Motivation: we should construct this in a way to minimize nested lambdas to inline as much as possible!
	*/
    template<typename Tnumeric>
    auto get_complex_path(const std::complex<double> split_point, const Tnumeric y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const datatypes::complex_root complex_root, const std::complex<double> sing_point) -> std::tuple< path_function, path_function> {

        auto C = std::real(complex_root.c);
        auto rc = std::real(complex_root.c);
        auto ic = std::imag(complex_root.c);

        auto sqc = std::sqrt(complex_root.c_0);

        auto cTimesCconj = rc * rc + ic * ic;
        //(const double x, const double y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r) 
        auto S = 1.0 / (complex_root.c_0 - q * q);
        auto Psp = math_utils::calculate_P_x(split_point, y, A, b, r);
        // K = @(t) sqrt(PsP)+q.*sP+1i.*t;
        auto K = [=](const auto t) -> auto {
            return std::sqrt(Psp) + q * split_point + 1.i * t;
        };//generate_K_x(split_point, Psp, q);
        auto T = [=](const auto t) -> auto {
            return (q * K(t) - complex_root.c_0 * C) * S;
        };
        path_function path;
        //observation:
        // two cases q == sqc or q == -sqc
        // in either case we coud just unify these branches because 
        // all that changes in the path function is the sign in the denominator and U, indicating that abs(sqc) may be sufficient
        if (std::abs(q - sqc) <= std::numeric_limits<double>::epsilon()) {
            auto U = [=](const double t) -> auto {
                return std::sqrt(Psp) + sqc * split_point + 1i * t;
            };
            path = [=](const double t) -> auto {
                return (U(t) * U(t) - cTimesCconj * complex_root.c_0) / (2. * (U(t) * sqc - complex_root.c_0 * C));
            };
        }
        else if (std::abs(q + sqc) <= std::numeric_limits<double>::epsilon()) {
            auto U = [&Psp, &sqc, &split_point](const double t) -> auto {
                return std::sqrt(Psp) - sqc * split_point + 1i * t;
            };
            path = [=](const double t) -> auto {
                return (U(t) * U(t) - cTimesCconj * complex_root.c_0) / (2. * (-1. * U(t) * sqc - complex_root.c_0 * C));
            };
        }

        else if (std::abs(q) > sqc) {
            auto sign = q > 0 ? -1. : 1.;
            path = [=](const double t) -> auto {
                return sign * std::sqrt((K(t) * K(t) - cTimesCconj * complex_root.c_0) * S + T(t) * T(t)) - T(t);
            };
        }

        else if (std::abs(q) < sqc) {
            auto sign = std::real(split_point) >= std::real(sing_point) ? 1. : -1.;
            path = [=](const double t) -> auto {
                return sign * std::sqrt((std::pow(K(t), 2.) - cTimesCconj * complex_root.c_0) * S + T(t) * T(t)) - T(t);
            };
        }

        else if (std::abs(q) <= std::numeric_limits<double>::epsilon()) {
            auto path_fun = [=](const double t) -> auto {
                return std::sqrt((Psp + 2. * 1.i * std::sqrt(Psp) * t - t * t) * 1. / complex_root.c_0 - (std::imag(complex_root.c) * std::imag(complex_root.c))) + C;
            };
            if (std::real(split_point) < C) {
                path = [=](const double t) -> auto {
                    return -path_fun(t) + 2 * C;
                };
            }
            else
            {
                path = path_fun;
            }
        }

        path_function path_derivative;
        auto Pp = [=](const double t) -> auto {
            return  math_utils::calculate_P_x(path(t), y, A, b, r);
        };

        path_derivative = [=](const double t) -> auto {
            return std::sqrt(Pp(t)) * 1.i / (complex_root.c_0 * (path(t) - C) + q * std::sqrt(Pp(t)));
        };

        return std::make_tuple(path, path_derivative);
    }

		
	auto get_weighted_path_1d(const std::complex<double> split_point, const double y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	auto get_weighted_path(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point) ->path_function;
	auto get_weighted_path_2d(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	//auto get_weighted_path_2d(const std::complex<double> split_point, const double y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	auto get_weighted_path_y(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
}