#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <steepest_descent/integrator.h>
#include <steepest_descent/integral.h>

#include <complex>
#include <vector>
#include <armadillo>

namespace py = pybind11;
using namespace std::complex_literals;
//https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html

//double integrate_1d_bulk(std::vector<Eigen::Ref<MatrixType>>)

Eigen::dcomplex integrate_1d(Eigen::Ref<Eigen::Matrix<double, 3,2>> A1, Eigen::Ref<Eigen::Vector3d> b1, Eigen::Ref<Eigen::Vector3d>, Eigen::Ref<Eigen::Vector3d> mu1, double y, int k, double left_split, double right_split) {
	
	arma::mat  A{ {0, 0}, {1, 0}, {1, 1 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };

	arma::vec mu{ 1,-0.5, -0.5 };
	//auto y = 0.;
	//auto k = 100;
	//auto left_split = 0.;
	//auto right_split = 1.;

	auto integrator = [](const integrator::integrand_fun fun, const std::complex<double> first_split_point, const std::complex<double> second_split_point) -> std::complex<double> {
		return  -0.090716 + 0.259133i;
	};
	integral::integral_1d integral1d(k, integrator, 0.1);
	auto result = integral1d(A, b, r, mu, y, left_split, right_split);
	std::cout << "result: " << result << std::endl;
	return result;
}
