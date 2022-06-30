#pragma once

#include "math_utils.h"
#include <complex>

namespace path_utils {

	using namespace std::literals::complex_literals;
	typedef std::function <const std::complex<double>(const double t)> path_function;


	auto generate_K_x(const std::complex<double> x, const std::complex<double> P_x, const double q)->std::function<const std::complex<double>(const double t)>;

	template<typename T>
	auto get_complex_path(const std::complex<double> split_point, const std::function<const T(const double x)> function_Px, const double q, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->std::tuple<path_function, path_function>
	{
        auto C = std::real(complex_root.c);
        auto c_real_squared = std::pow(C, 2);
        auto rc = std::real(complex_root.c);
        auto ic = std::imag(complex_root.c);

        auto sqc = std::sqrt(complex_root.c_0);

        auto cTimesCconj = rc * rc + ic * ic; // |c|^2


        //For chapter 4.1 formular 12 K^1_hx and K^2_hx
        auto S = 1.0 / (complex_root.c_0 - q * q);

        auto Psp = function_Px(split_point);

        auto K = [=](const auto t) -> auto {
            return std::sqrt(Psp) + q * split_point + 1.i * t;
        };

        // for chaper 4.1 formular 12 K^2_hx
        auto K_2 = [=](const auto t) -> auto {
            return (q * K(t) - complex_root.c_0 * C) * S;
        };

        path_function path;
        if (std::abs(q - sqc) <= std::numeric_limits<double>::epsilon()) {
            // sqc = q
            path = [=](const double t) -> auto {
                auto upper_term = (K(t) * K(t) - cTimesCconj * complex_root.c_0);
                auto lower_term = (2. * (K(t) * sqc - complex_root.c_0 * C));
                return  upper_term  / lower_term;
            };
        }
        else if (std::abs(q + sqc) <= std::numeric_limits<double>::epsilon()) {
            // sqc = -q
            path = [=](const double t) -> auto {
                auto upper_term = (K(t) * K(t) - cTimesCconj * complex_root.c_0);
                auto lower_term = (-2. * (K(t) * sqc - complex_root.c_0 * C));
                return  upper_term / lower_term;
            };
        }

        else if (std::abs(q) > sqc) {
            auto sign = q > 0 ? -1. : 1.;
            // K^1_hx +/- K^2_hx => umgestellt mit T (das ist K^1_hx mit negativem Vorzeichen!
            path = [=](const double t) -> auto {
                return sign * std::sqrt((K(t) * K(t) - cTimesCconj * complex_root.c_0) * S + K_2(t) * K_2(t)) - K_2(t);
            };
        }

        else if (std::abs(q) < sqc) {
            auto sign = std::real(split_point) >= std::real(sing_point) ? 1. : -1.;
            path = [=](const double t) -> auto {
                return sign * std::sqrt((std::pow(K(t), 2.) - cTimesCconj * complex_root.c_0) * S + K_2(t) * K_2(t)) - K_2(t);
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
            return function_Px(path(t));
        };

        path_derivative = [=](const double t) -> auto {
            return std::sqrt(Pp(t)) * 1.i / (complex_root.c_0 * (path(t) - C) + q * std::sqrt(Pp(t)));
        };

        return std::make_tuple(path, path_derivative);
	}


	auto get_complex_path(const std::complex<double> split_point, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, const double q, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->std::tuple<path_function, path_function>;


    auto get_weighted_path(const std::complex<double> split_point, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
  
    auto get_weighted_path_1d(const double k, const std::complex<double> split_point, const double q, const double s, const std::function<const std::complex<double>(const std::complex<double> x)> function_Px, path_function path, path_function derivative)->path_function;

}