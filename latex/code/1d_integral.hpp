namespace integral {
	using namespace std::complex_literals;

	struct integral_1d {
		integral_1d(const config::configuration config, integrator::gsl_integrator* integrator);
		integral_1d(const config::configuration config, integrator::gsl_integrator* integrator, const std::vector<double> nodes, const std::vector<double> weights);
		integral_1d(const config::configuration& config, integrator::gsl_integrator* integrator, const std::vector<double> nodes, const std::vector<double> weights, const path_utils::path_function_generator path_function_generator, math_utils::green_fun_generator green_fun_generator);


		auto operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const arma::vec3& theta, const double y, const double left_split, const double right_split) const->std::complex<double>;
		auto operator()(const arma::mat& A, const arma::vec3& b, const arma::vec3& r, const double q, const std::complex<double> s, const std::complex<double> y, const std::complex<double> c, const double c_0, const double left_split, const double right_split) const->std::complex<double>;

	private:
		config::configuration config;
		integrator::gsl_integrator* integrator;
		std::vector<double> nodes, weights;

		path_utils::path_function_generator path_function_generator;
		math_utils::green_fun_generator green_fun_generator;
	};
}