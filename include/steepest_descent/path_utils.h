#pragma once

#include "math_utils.h"
#include <armadillo>
#include <complex>

namespace path_utils {

	using namespace std::literals::complex_literals;
	typedef std::function <const std::complex<double>(const double t)> path_function;
    typedef std::function<path_function(const std::complex<double> split_point, const std::complex<double> y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)> path_function_generator;

    template<typename Tnumeric>
    auto get_complex_path(const std::complex<double>& split_point, const Tnumeric Px, const double& q, const datatypes::complex_root& complex_root, const std::complex<double>& sing_point) -> path_function {
        auto C = std::real(complex_root.c);
        auto rc = std::real(complex_root.c);
        auto ic = std::imag(complex_root.c);

        auto sqc = std::sqrt(complex_root.c_0);

        auto sqrtPx = std::sqrt(Px);
        auto cTimesCconj = rc * rc + ic * ic;
        //chapter 4.1 
        auto S = 1.0 / (complex_root.c_0 - q * q);
        auto K = [=](const auto t) -> auto {
            return sqrtPx + q * split_point + 1.i * t;
        };
        auto T = [=](const auto t) -> auto {
            return (q * K(t) - complex_root.c_0 * C) * S;
        };
        path_function path;
        //observation:
        // two cases q == sqc or q == -sqc
        // in either case we coud just unify these branches because 
        // all that changes in the path function is the sign in the denominator and U, indicating that abs(sqc) may be sufficient
        //case q = +/- sqrt(c_0)
        if (std::abs(q - sqc) <= std::numeric_limits<double>::epsilon()) {
            path = [=](const double t) -> auto {
                return (K(t) * K(t) - cTimesCconj * complex_root.c_0) / (2. * (K(t) * sqc - complex_root.c_0 * C));
            };
        }
        else if (std::abs(q + sqc) <= std::numeric_limits<double>::epsilon()) {
            path = [=](const double t) -> auto {
                return (K(t) * K(t) - cTimesCconj * complex_root.c_0) / (2. * (-1. * K(t) * sqc - complex_root.c_0 * C));
            };
        }
        //case |q| < sqrt(c_0)
        else if (std::abs(q) < sqc) {
            auto sign = std::real(split_point) >= std::real(sing_point) ? 1. : -1.;
            path = [=](const double t) -> auto {
                return sign * std::sqrt((std::pow(K(t), 2.) - cTimesCconj * complex_root.c_0) * S + T(t) * T(t)) - T(t);
            };

        }
        //case |q| > sqrt(c_0)
        else if (std::abs(q) > sqc) {
            auto sign = q > 0 ? -1. : 1.;
            path = [=](const double t) -> auto {
                return sign * std::sqrt((K(t) * K(t) - cTimesCconj * complex_root.c_0) * S + T(t) * T(t)) - T(t);
            };
        }


        //what case is this? q == 0 was the first case in matlab code. maybe an optimization?
        else if (std::abs(q) <= std::numeric_limits<double>::epsilon()) {
            auto path_fun =  [=](const double t) -> auto {
                return std::sqrt((Px + 2. * 1.i * sqrtPx * t - t * t) * 1. / complex_root.c_0 - (std::imag(complex_root.c) * std::imag(complex_root.c))) + C;
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

        return path;
    }

    template<typename Tnumeric>
    inline auto get_path_derivative(const path_function & path, const Tnumeric y, const datatypes::matrix & A, const  arma::vec3 & b, const  arma::vec3 & r, const double& q, const datatypes::complex_root& complex_root) -> path_function {
        auto C = std::real(complex_root.c);

        //chapter   4.2 Equation 15
        auto Pp = [=](const double t) -> auto {
            return  math_utils::calculate_P_x(path(t), y, A, b, r);
        };

        return [=](const double t) -> auto {
            return std::sqrt(Pp(t)) * 1.i / (complex_root.c_0 * (path(t) - C) + q * std::sqrt(Pp(t)));
        };
    }

	auto get_weighted_path_1d(const std::complex<double> split_point, const double y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	auto get_weighted_path(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point) ->path_function;
	auto get_weighted_path_2d(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	auto get_weighted_path_y(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
}