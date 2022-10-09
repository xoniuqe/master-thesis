template<class integrand_fun>
auto operator()(integrand_fun integrand, const std::complex<double> x_start, const std::complex<double> x_end, const double y_start, const double y_end) const ->std::complex<double>
{
    double result_real, result_imag;

    auto f_real = make_gsl_function([&](double x) {
        double inner_result;
        auto inner = make_gsl_function([&](double y) {
            return std::real(integrand(x, y));
            });
        gsl_integration_cquad(&inner, y_start, y_end, 1e-16, 1e-6, inner_workspace, &inner_result, NULL, NULL);
        return inner_result;
        });
    

    auto status = gsl_integration_cquad(&f_real, std::real(x_start), std::real(x_end), 1e-16, 1e-6, workspace, &result_real, NULL, NULL);
[...]
