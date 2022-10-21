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
		gsl_integrator_2d(const size_t n=4000) : n(n) {
			workspace = gsl_integration_workspace_alloc(n);
			inner_workspace = gsl_integration_workspace_alloc(n);
		}

		gsl_integrator_2d(gsl_integrator_2d& other) = delete;

		gsl_integrator_2d(gsl_integrator_2d&& other)
		{
			std::lock_guard<std::mutex> guard(other.integration_mutex);
			inner_workspace = other.inner_workspace;
			workspace = other.workspace;
			other.inner_workspace = nullptr;
			other.workspace = nullptr;
			n = other.n;
		}

		~gsl_integrator_2d() {
			gsl_integration_workspace_free(inner_workspace);
			gsl_integration_workspace_free(workspace);
		}

		template<class integrand_fun>
		auto operator()(integrand_fun integrand, const std::complex<double> x_start, const std::complex<double> x_end, const double y_start, const double y_end) const ->std::complex<double>
		{
			std::lock_guard<std::mutex> guard(integration_mutex);

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


			auto f_imag = make_gsl_function([&](double x) {
				double inner_result;
				auto inner = make_gsl_function([&](double y) {
					return std::imag(integrand(x, y));
					});
				gsl_integration_qags(&inner, y_start, y_end, 1e-12, 1e-12, n, inner_workspace, &inner_result, &abs_error_imag_inner);
				return inner_result;
				});
			gsl_integration_qags(&f_imag, std::real(x_start), std::real(x_end), 1e-12, 1e-12, n, workspace, &result_imag, &abs_error_real);
			return std::complex<double>(result_real, result_imag);
		}
	private:
		mutable std::mutex integration_mutex;
		size_t n;
		gsl_integration_workspace* workspace, * inner_workspace;

	};
}
