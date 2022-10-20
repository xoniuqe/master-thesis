integrator::gsl_integrator integrator_1d;
config::configuration config1d;
config1d.wavenumber_k = config.wavenumber_k;
config1d.tolerance = config.tolerance;

// zu integrierende Funktion für GSL
math_utils::green_fun_generator fun_gen = [](const double& k, const std::complex<double>& y, const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double& q, const std::complex<double>& s) -> math_utils::green_fun
{
    return [&](const double x) -> auto {
        auto Px = math_utils::calculate_P_x(x, y, A, b, r);
        auto sqrtPx = std::sqrt(Px);
        auto res = std::exp(1.i * k * (sqrtPx + q * x + s));
        return res;
    };
};
integral_1d_test partial_integral(config1d, &integrator_1d, nodes, weights, path_utils::get_weighted_path_y, fun_gen);