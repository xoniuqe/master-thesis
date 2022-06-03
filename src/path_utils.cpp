#include <steepest_descent/path_utils.h>
#include <steepest_descent/math_utils.h>

#include <armadillo>

#include <limits>
#include <cmath>
#include <complex>


#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif

namespace path_utils {

    using namespace std::literals::complex_literals;


    auto generate_K_x(const std::complex<double> x, const std::complex<double> P_x, const double q) -> std::function<const std::complex<double>(const double t)> {
        return[=](const double t) -> auto {
            //auto P_x = c_0 * x * x + c_0 * std::abs(c) * std::abs(c) - 2. * c_0 * x * std::real(c);
            return std::sqrt(P_x) + q * x + t * 1i;
        };
    }

    //function[cPath,ddtcPath] = GetComplexPath(sP,y,A,b,r,q,c,c_0,singPoint)

    auto get_complex_path(const std::complex<double> split_point, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, const double q, const datatypes::complex_root complex_root, const std::complex<double> sing_point) -> std::tuple< path_function, path_function>{
        /*

    Cb = conj(c);
    sqc = sqrt(c_0);
    C = real(c);

    S = 1/(c_0-q^2);
    PsP = Px(sP,y,A,b,r);
    K = @(t) sqrt(PsP)+q.*sP+1i.*t;
    T = @(t) (q.*K(t)-c_0*real(c))*S;
    */

        auto C = std::real(complex_root.c);
        auto c_real_squared = std::pow(C, 2);
        auto rc = std::real(complex_root.c);
        auto ic = std::imag(complex_root.c);

        auto sqc = std::sqrt(complex_root.c_0);

        auto cTimesCconj = rc * rc + ic * ic;
        //(const double x, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r) 
        auto S = 1.0 / (complex_root.c_0 - q * q);
        auto Psp = math_utils::calculate_P_x(split_point, y, A, b, r);
        // K = @(t) sqrt(PsP)+q.*sP+1i.*t;
        auto K = [=](const auto t) -> auto {
            return std::sqrt(Psp) + q * split_point + 1.i * t;
        };//generate_K_x(split_point, Psp, q);
        auto T = [=](const auto t) -> auto {
            return (q * K(t) - complex_root.c_0 * C) * S;
        };
        std::cout << "K(1)" << K(1.) << std::endl;

        path_function path;
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
                return sign * std::sqrt( (K(t) * K(t) - cTimesCconj * complex_root.c_0) * S + T(t) * T(t)) - T(t);
            };
        }

        else if (std::abs(q) < sqc) {
            auto sign = std::real(split_point) >= std::real(sing_point) ? 1. : -1.;
            path = [=](const double t) -> auto {
                return sign * std::sqrt((std::pow(K(t),2.) - cTimesCconj * complex_root.c_0) * S + T(t) * T(t)) - T(t);
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
            return std::sqrt(Pp(t)) * 1.i /(complex_root.c_0 * (path(t) - C) + q * std::sqrt(Pp(t)));
        };
   /*

%Derivative of the path obtained by injecting the explicit formulas
%into the ODE.

Pp = @(t) Px(cPath(t),y,A,b,r);
ddtcPath = @(t) (Pp(t)).^(1/2)*1i./(c_0*(cPath(t)-  C)+q*(Pp(t)).^(1/2));*/

        return std::make_tuple(path, path_derivative);
    }

    auto get_weighted_path(const std::complex<double> split_point, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function {
        auto [path, derivative] = get_complex_path(split_point, y, A, b, r, q, complex_root, sing_point);
       /* std::cout << "split: " << split_point << std::endl;
        std::cout << "path(1) " << path(1) << std::endl;
        std::cout << "derivative(1) " << derivative(1) << std::endl;

        std::cout << " ddtcPath(1 / k): " << derivative(1 / k) << std::endl;
        std::cout << "  (Px(sP,y,A,b,r).^(1/2) " << std::sqrt(math_utils::calculate_P_x(split_point, y, A, b, r)) << std::endl;
        std::cout << "q: " << q << std::endl;
        std::cout << "sP: " << split_point << std::endl;
        std::cout << "s: " << s << std::endl;

        std::cout << " q*sP+s " << q * split_point + s << std::endl;
        std::cout << " (Px(sP,y,A,b,r).^(1/2)+q*sP+s) " << (std::sqrt(math_utils::calculate_P_x(split_point, y, A, b, r)) + q * split_point + s) << std::endl;
        std::cout << " exp(1i*k*(Px(sP,y,A,b,r).^(1/2)+q*sP+s)) " << std::exp(1i * k * (std::sqrt(math_utils::calculate_P_x(split_point, y, A, b, r)) + q * split_point + s)) << std::endl;
        std::cout << " Px(cPath(t/k),y,A,b,r).^(-1/2): " << (1. / std::sqrt(math_utils::calculate_P_x(path(1 / k), y, A, b, r))) << std::endl;*/
        //
        //    weightFunPerPath = @(t) ddtcPath(t/k).*exp(1i*k*(Px(sP,y,A,b,r).^(1/2)+q*sP+s)).*k^(-1).*Px(cPath(t/k),y,A,b,r).^(-1/2);  
        //@(t) ddtcPath(t/k).*exp(1i*k*(Px(sP,y,A,b,r).^(1/2)+q*sP+s)).*k^(-1).*Px(cPath(t/k),y,A,b,r).^(-1/2);
        return [=](const double t) -> auto {
            return derivative(t / k) * std::exp(1i * k * (std::sqrt(math_utils::calculate_P_x(split_point, y, A, b, r)) + q * split_point + s)) * (1. / k) * (1. /std::sqrt(math_utils::calculate_P_x(path(t / k), y, A, b, r)));
        };
    }
}