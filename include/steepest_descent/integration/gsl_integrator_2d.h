#pragma once

#include <functional>
#include <complex>

#include "gsl_function_wrapper.h"

#include <gsl/gsl_complex.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_laguerre.h>
#include <iostream>
#include <mutex>


namespace integrator {
	using namespace std::complex_literals;



	struct gsl_integrator_2d {
		gsl_integrator_2d(const size_t n=1000) {
			workspace = gsl_integration_cquad_workspace_alloc(n);
			inner_workspace = gsl_integration_cquad_workspace_alloc(n);
		}

		gsl_integrator_2d(gsl_integrator_2d& other) = delete;

		~gsl_integrator_2d() {
			gsl_integration_cquad_workspace_free(inner_workspace);
			gsl_integration_cquad_workspace_free(workspace);
		}

		template<class integrand_fun>
		auto operator()(integrand_fun integrand, const std::complex<double> x_start, const std::complex<double> x_end, const double y_start, const double y_end) const ->std::complex<double>
		{
			std::lock_guard<std::mutex> guard(integration_mutex);

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

			if (status > 0) {
				std::cout << "error" << std::endl;
			}
			auto f_imag = make_gsl_function([&](double x) {
				double inner_result;
				auto inner = make_gsl_function([&](double y) {
					return std::imag(integrand(x, y));
					});
				gsl_integration_cquad(&inner, y_start, y_end, 1e-16, 1e-6, inner_workspace, &inner_result, NULL, NULL);
				return inner_result;
				});


			auto status2 = gsl_integration_cquad(&f_imag, std::real(x_start), std::real(x_end), 1e-16, 1e-6, workspace, &result_imag, NULL, NULL);


			if (status2 > 0) {
				std::cout << "error2" << std::endl;
			}
			return std::complex<double>(result_real, result_imag);
		}
	private:
		mutable std::mutex integration_mutex;
		gsl_integration_cquad_workspace *workspace, *inner_workspace;
	};
}
