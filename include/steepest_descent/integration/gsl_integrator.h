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



	struct gsl_integrator {
		gsl_integrator(const size_t n = 2000) : n(n) {
			workspace = gsl_integration_workspace_alloc(n);

		}

		gsl_integrator(gsl_integrator& other) = delete;

		~gsl_integrator() {
			gsl_integration_workspace_free(workspace);
		}

		template<class integrand_fun>
		auto operator()(integrand_fun integrand, const std::complex<double> first_split_point, const std::complex<double> second_split_point) const ->std::complex<double>
		{
			std::lock_guard<std::mutex> guard(integration_mutex);
			double result_real, result_imag;
			double res_re2, res_im2;
			double abs_error_real, abs_error_imag;
			size_t evals_real, evals_imag;
			gsl_function F_real;
			F_real.function = [](double x, void* p)->double {
				return std::real((*static_cast<integrand_fun*>(p))(x));
			};
			F_real.params = &integrand;
			gsl_integration_qags(&F_real, std::real(first_split_point), std::real(second_split_point), 1e-12, 1e-12, n, workspace, &result_real, &abs_error_real);

			gsl_function F_imag;
			F_imag.function = [](double x, void* p)->double {
				return std::imag((*static_cast<integrand_fun*>(p))(x));
			};
			F_imag.params = &integrand;
			gsl_integration_qags(&F_imag, std::real(first_split_point), std::real(second_split_point), 1e-12, 1e-12, n, workspace, &result_imag, &abs_error_imag);
			return std::complex<double>(result_real , result_imag );
		}
	private:
		mutable std::mutex integration_mutex;
		size_t n;
		gsl_integration_workspace* workspace;
	};
}