#include <steepest_descent/path_utils.h>
#include <steepest_descent/math_utils.h>

#include <armadillo>

#include <limits>
#include <cmath>
#include <complex>


#ifdef _WIN32
//#include <corecrt_math_defines.h>
#endif

namespace path_utils {

    using namespace std::literals::complex_literals;


    auto generate_K_x(const std::complex<double> x, const std::complex<double> P_x, const double q) -> std::function<const std::complex<double>(const double t)> {
        return[=](const double t) -> auto {
            return std::sqrt(P_x) + q * x + t * 1i;
        };
    }



    auto get_weighted_path_1d(const std::complex<double> split_point, const double y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function {
        auto Px = math_utils::calculate_P_x(split_point, y, A, b, r);
        auto path = get_complex_path(split_point, Px, q, complex_root, sing_point);
        auto derivative = get_path_derivative(path, y, A, b, r, q, complex_root);
        return [=](const double t) -> auto {
            return derivative(t / k) * std::exp(1i * k * (std::sqrt(math_utils::calculate_P_x(split_point, y, A, b, r)) + q * split_point + s)) * (1. / k) * (1. /std::sqrt(math_utils::calculate_P_x(path(t / k), y, A, b, r)));
        };
    }

    auto get_weighted_path(const std::complex<double> split_point, const std::complex<double> y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function {
        auto Px = math_utils::calculate_P_x(split_point, y, A, b, r);
        auto path = get_complex_path(split_point, Px, q, complex_root, sing_point);
        auto derivative = get_path_derivative(path, y, A, b, r, q, complex_root);
        return [=](const double t) -> auto {
            return derivative(t / k) * std::exp(1i * k * (std::sqrt(math_utils::calculate_P_x(split_point, y, A, b, r)) + q * split_point + s)) * (1. / k) * (1. / std::sqrt(math_utils::calculate_P_x(path(t / k), y, A, b, r)));
        };
    }

    auto get_weighted_path_2d(const std::complex<double> split_point, const std::complex<double> y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function {
       // auto [path, derivative] = get_complex_path(split_point, y, A, b, r, q, complex_root, sing_point);
        auto Px = math_utils::calculate_P_x(split_point, y, A, b, r);
        auto path = get_complex_path(split_point, Px, q, complex_root, sing_point);
        auto derivative = get_path_derivative(path, y, A, b, r, q, complex_root);
        return [=](const double t) -> auto {
            return derivative(t / k) * (1. / k) * (1. / std::sqrt(math_utils::calculate_P_x(path(t / k), y, A, b, r)));
        }; // Equation (2.2) from APPLYING THE NUMERICAL METHOD OF STEEPEST DESCENT ON MULTIVARIATE OSCILLATORY INTEGRALS IN SCATTERING THEORY?
    }

    auto get_weighted_path_y(const std::complex<double> split_point, const std::complex<double> y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function {
       // auto [path, derivative] = get_complex_path(split_point, y, A, b, r, q, complex_root, sing_point);
        auto Px = math_utils::calculate_P_x(split_point, y, A, b, r);
        auto path = get_complex_path(split_point, Px, q, complex_root, sing_point);
        auto derivative = get_path_derivative(path, y, A, b, r, q, complex_root);

        return [=](const double t) -> auto {
            return derivative(t / k) * (1. / k) * std::exp(1.i * k * (std::sqrt(Px) + q * split_point + s));
        };
    }


}