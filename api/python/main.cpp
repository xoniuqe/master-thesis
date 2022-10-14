#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <steepest_descent/integral_1d.h>
#include <steepest_descent/integral_2d.h>
#include <steepest_descent/gauss_laguerre.h>

#include <vector>
#include <armadillo>

namespace py = pybind11;
using namespace std::complex_literals;

//https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html

// do not forget to install numpy!



typedef Eigen::Matrix<double, 3, 2> MatrixD;
typedef Eigen::Matrix<int, 3, 2> MatrixI;

Eigen::dcomplex integrate_1d(MatrixD A1, Eigen::Vector3d b1, Eigen::Vector3d r1, Eigen::Vector3d theta1, double y, int k, double left_split, double right_split, const std::vector<double> nodes, const std::vector<double> weights) {
	
	arma::mat A(A1.data(), 3, 2, false, false);
	arma::vec b(b1.data(), 3);
	arma::vec r(r1.data(), 3);
	arma::vec theta(theta1.data(), 3);

	
	config::configuration config;
	config.wavenumber_k = k;
	config.tolerance = 0.1;
	integrator::gsl_integrator gslintegrator;

	integral::integral_1d integral1d(config, &gslintegrator);

	return integral1d(A, b, r, theta, y, left_split, right_split);
}



Eigen::dcomplex integrate2d(MatrixD triangle_A,  Eigen::Vector3d triangle_b, Eigen::Vector3d observation_point, Eigen::Vector3d direction, const std::vector<double> nodes, const std::vector<double> weights, int k, double resolution = 0.1)
{
	arma::mat A(triangle_A.data(), 3, 2, false, false);
	arma::vec b(triangle_b.data(), 3);
	arma::vec r(observation_point.data(), 3);;
	arma::vec theta(direction.data(), 3);

	config::configuration_2d config;
	config.wavenumber_k = k;
	config.tolerance = 0.1;
	config.y_resolution = resolution;
	integrator::gsl_integrator_2d gslintegrator_2d;
	integrator::gsl_integrator gslintegrator;


	integral::integral_2d integral(config, &gslintegrator, &gslintegrator_2d, nodes, weights);
	return integral(A, b, r, theta);
}	


Eigen::dcomplex integrate2d_config(MatrixD triangle_A, Eigen::Vector3d triangle_b, Eigen::Vector3d observation_point, Eigen::Vector3d direction, const std::vector<double> nodes, const std::vector<double> weights, const config::configuration_2d config)
{
	arma::mat A(triangle_A.data(), 3, 2, false, false);
	arma::vec b(triangle_b.data(), 3);
	arma::vec r(observation_point.data(), 3);;
	arma::vec theta(direction.data(), 3);

	integrator::gsl_integrator_2d gslintegrator_2d;
	integrator::gsl_integrator gslintegrator;


	integral::integral_2d integral(config, &gslintegrator, &gslintegrator_2d, nodes, weights);
	return integral(A, b, r, theta);
}



PYBIND11_MODULE(stedepy, m) {
	m.doc() = "integrator test plugin";
	m.def("integrate_1d", &integrate_1d, "Calculates the 1D complex integral");  
	m.def("integrate_2d", &integrate2d, "Calculates the 2d integral");
	m.def("integrate_2d", &integrate2d_config, "Calculates the 2d integral with config param");

	m.def("calculate_laguerre_points", [](const int n) -> auto {
		return gauss_laguerre::calculate_laguerre_points_and_weights(n);
	}, "Returns 'n' laguerre nodes and weights");
	m.def("test", [](const std::vector<double> nodes) {
		for (auto& node : nodes) {
			std::cout << node << "\n";
		}
		});

	py::class_<config::configuration_2d> config(m, "configuration");

	config.def(py::init<>())
		.def_readwrite("tol", &config::configuration_2d::tolerance)
		.def_readwrite("k", &config::configuration_2d::wavenumber_k)
		.def_readwrite("res", &config::configuration_2d::y_resolution);
}