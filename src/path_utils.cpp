#include <steepest_descent/path_utils.h>
#include <steepest_descent/math_utils.h>

#include <armadillo>

#include <limits>
#include <cmath>
#include <complex>

namespace path_utils {

    using namespace std::literals::complex_literals;

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
        auto Px = math_utils::calculate_P_x(split_point, y, A, b, r);
        auto path = get_complex_path(split_point, Px, q, complex_root, sing_point);
        auto derivative = get_path_derivative(path, y, A, b, r, q, complex_root);
        return [=](const double t) -> auto {
            return derivative(t / k) * (1. / k) * (1. / std::sqrt(math_utils::calculate_P_x(path(t / k), y, A, b, r)));
        }; // Equation (2.2) from APPLYING THE NUMERICAL METHOD OF STEEPEST DESCENT ON MULTIVARIATE OSCILLATORY INTEGRALS IN SCATTERING THEORY?
    }

    auto get_weighted_path_y(const std::complex<double> split_point, const std::complex<double> y, const datatypes::matrix& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function {
        auto Px = math_utils::calculate_P_x(split_point, y, A, b, r);
        auto path = get_complex_path(split_point, Px, q, complex_root, sing_point);
        auto derivative = get_path_derivative(path, y, A, b, r, q, complex_root);

        return [=](const double t) -> auto {
            return derivative(t / k) * (1. / k) * std::exp(1.i * k * (std::sqrt(Px) + q * split_point + s)); //equation 28
        };
    }


}