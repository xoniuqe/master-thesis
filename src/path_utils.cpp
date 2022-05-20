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


    auto generate_K_x(const double x, const double P_x, const double q) -> std::function<const std::complex<double>(const double t)> {
        return[&](const double t) -> auto {
            //auto P_x = c_0 * x * x + c_0 * std::abs(c) * std::abs(c) - 2. * c_0 * x * std::real(c);
            return std::sqrt(P_x) + q * x + t * 1i;
        };
    }

    //function[cPath,ddtcPath] = GetComplexPath(sP,y,A,b,r,q,c,c_0,singPoint)

    auto get_complex_path(const double split_point, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, const double q, const datatypes::complex_root complex_root) -> std::tuple< path_function, path_function>{
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
        auto K = generate_K_x(split_point, Psp, q);
        auto T = [&](const auto t) -> auto {
            return (q * K(t) - complex_root.c_0 * C) * S;
        };

        path_function path;
        if (std::abs(q - sqc) <= std::numeric_limits<double>::epsilon()) {
            auto U = [&](const std::complex<double> t) -> auto {
                return std::sqrt(Psp) + sqc * split_point + 1i * t;
            };
            path = [&](const std::complex<double> t) -> auto {
                return (U(t) * U(t) - cTimesCconj * complex_root.c_0) / (2. * (U(t) * sqc - complex_root.c_0 * C));
            };
        }
        if (std::abs(q + sqc) <= std::numeric_limits<double>::epsilon()) {
            auto U = [&Psp, &sqc, &split_point](const std::complex<double> t) -> auto {
                return std::sqrt(Psp) - sqc * split_point + 1i * t;
            };
            path = [&](const std::complex<double> t) -> auto {
                return (U(t) * U(t) - cTimesCconj * complex_root.c_0) / (2. * (-1. * U(t) * sqc - complex_root.c_0 * C));
            };
        }
        /*
     if q == 0;
    cPath = @(t) ((PsP+2*1i*PsP^(1/2).*t-t.^2)*1/c_0-imag(c)^2).^(1/2)+C;
    if sP < C;
        cPath = @(t) -cPath(t)+2*C;
    end
end

if q == sqc;
    U = @(t) sqrt(PsP)+sqc*sP+1i.*t;
    cPath = @(t) ((U(t)).^2-c_0*c*Cb)/2.*(U(t)*sqc-c_0*C).^(-1);
end

if q == -sqc;
    U = @(t) sqrt(PsP)-sqc*sP+1i.*t;
    cPath = @(t) ((U(t)).^2-c_0*c*Cb)/2.*(-U(t)*sqc-c_0*C).^(-1);
end

if q > 0 && abs(q) > sqc;
    cPath = @(t) -((K(t).^2-c_0*c*Cb)*S+T(t).^2).^(1/2)-T(t);
end

if q < 0 && abs(q) > sqc;
    cPath = @(t) ((K(t).^2-c_0*c*Cb)*S+T(t).^2).^(1/2)-T(t);
end

if abs(q) < sqc;
    if sP >= singPoint;
        cPath = @(t) ((K(t).^2-c_0*c*Cb)*S+T(t).^2).^(1/2)-T(t);
    else
        cPath = @(t) -((K(t).^2-c_0*c*Cb)*S+T(t).^2).^(1/2)-T(t);
    end
end


%Derivative of the path obtained by injecting the explicit formulas
%into the ODE.

Pp = @(t) Px(cPath(t),y,A,b,r);
ddtcPath = @(t) (Pp(t)).^(1/2)*1i./(c_0*(cPath(t)-C)+q*(Pp(t)).^(1/2));*/

        return std::make_tuple(path, [](const auto t) -> auto { return t; });
    }

}