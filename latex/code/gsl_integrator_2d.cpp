template<class integrand_fun>
auto operator()(integrand_fun integrand, const std::complex<double> x_start, const std::complex<double> x_end, const double y_start, const double y_end) const ->std::complex<double>
{
    double result_real, result_imag;
	double abs_error_real_inner, abs_error_real, abs_error_imag_inner, abs_error_imag;

    auto f_real = make_gsl_function([&](double x) {
        double inner_result;
        auto inner = make_gsl_function([&](double y) {
            return std::real(integrand(x, y));
            });
		gsl_integration_qags(&inner, y_start, y_end, 1e-12 , 1e-12, n, inner_workspace, &inner_result, &abs_error_real_inner);
        return inner_result;
        });
    

		gsl_integration_qags(&f_real, std::real(x_start), std::real(x_end), 1e-12, 1e-12, n, workspace, &result_real, &abs_error_real);
[...] // Berechnung des imaginären Teils analog.
