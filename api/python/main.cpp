#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <steepest_descent/integrator.h>
#include <steepest_descent/integral_1d.h>


//#include <complex>
#include <vector>
#include <armadillo>

namespace py = pybind11;
using namespace std::complex_literals;
//https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html

// do not forget to install numpy!

//double integrate_1d_bulk(std::vector<Eigen::Ref<MatrixType>>)

Eigen::dcomplex integrate_1d(Eigen::Matrix<double, 3,2> A1, Eigen::Vector3d b1,Eigen::Vector3d r1, Eigen::Vector3d mu1, double y, int k, double left_split, double right_split) {
	
	arma::mat A(A1.data(), 3, 2, false, false);
	//arma::mat A{ }
	//arma::mat  A{{0, 0}, {1, 0}, {1, 1}};

	arma::vec b(b1.data(), 3);// { 0, 0, 0 };

	arma::vec r(r1.data(), 3);// { 1, -0.5, -0.5 };

	arma::vec mu(mu1.data(), 3);// { 1, -0.5, -0.5 };

	
	config::configuration config;
	config.wavenumber_k = k;
	config.tolerance = 0.1;
	config.gauss_laguerre_nodes = 30;
	integrator::gsl_integrator gslintegrator;

	integral::integral_1d integral1d(config, &gslintegrator);

	return integral1d(A, b, r, mu, y, left_split, right_split);
	/*auto result = integral1d(A, b, r, mu, y, left_split, right_split);
	std::cout << "result: " << result << std::endl;
	return result;*/
	//return 1.0 + 3i;
}


PYBIND11_MODULE(stedepy, m) {
	m.doc() = "integrator test plugin";
	m.def("integrate_1d", &integrate_1d, "Calculates the 1D complex integral");   
}