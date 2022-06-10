#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <steepest_descent/integrator.h>
#include <steepest_descent/integral.h>

#include "gsl_function_wrapper.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

//#include <complex>
/*#include <vector>
/*#include <armadillo>*/

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

	
	auto integrator = [](const integrator::integrand_fun fun, const std::complex<double> first_split_point, const std::complex<double> second_split_point) -> std::complex<double> {
		//return  -0.090716 + 0.259133i;
		auto w
			= gsl_integration_cquad_workspace_alloc(1000);
		double result_real, result_imag;

		gsl_function_pp Fp_real([&fun](const double x) -> auto {
			auto res = std::real(fun(x));
			return res;
			});

		auto F_real = static_cast<gsl_function*>(&Fp_real);

		gsl_integration_cquad(F_real, std::real(first_split_point), std::real(second_split_point), 1e-10, 1e-6, w, &result_real, NULL, NULL);
		gsl_integration_cquad_workspace_free(w);
		w
			= gsl_integration_cquad_workspace_alloc(1000);

		gsl_function_pp Fp_imag([&fun](const double x) -> auto {
			return std::imag(fun(x));
			});

		auto F_imag = static_cast<gsl_function*>(&Fp_imag);
		gsl_integration_cquad(F_imag, std::real(first_split_point), std::real(second_split_point), 1e-10, 1e-6, w, &result_imag, NULL, NULL);
		gsl_integration_cquad_workspace_free(w);
		return result_real + result_imag * 1i;
	};
	
	integral::integral_1d integral1d(k, integrator, 0.1);

	return integral1d(A, b, r, mu, y, left_split, right_split);
	/*auto result = integral1d(A, b, r, mu, y, left_split, right_split);
	std::cout << "result: " << result << std::endl;
	return result;*/
	//return 1.0 + 3i;
}

int multiply(int i, int j) {
	return i * j + 1;
}

PYBIND11_MODULE(stedepy, m) {
	m.doc() = "integrator test plugin";
	m.def("integrate_1d", &integrate_1d, "Calculates the 1D complex integral");   
	m.def("multiply", &multiply, "A function which multiplies two numbers");
}