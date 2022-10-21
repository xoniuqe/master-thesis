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
			//workspace = gsl_integration_cquad_workspace_alloc(n);
			workspace = gsl_integration_workspace_alloc(n);

		}

		gsl_integrator(gsl_integrator& other) = delete;

		~gsl_integrator() {
			//gsl_integration_cquad_workspace_free(workspace);
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
			//  gsl_integration_qags (&F, 0, 1, 0, 1e-7, 1000,
			//w, & result, & error);
			//int gsl_integration_qng(const gsl_function *f, double a, double b, double epsabs, double epsrel, double *result, double *abserr, size_t *neval)
			//gsl_integration_qng(&F_real, std::real(first_split_point), std::real(second_split_point), 1e-10, 1e-10, &result_real, &abs_error_real, &evals_real);
			gsl_integration_qags(&F_real, std::real(first_split_point), std::real(second_split_point), 1e-12, 1e-12, n, workspace, &result_real, &abs_error_real);
				//auto status = gsl_integration_cquad(&F_real, std::real(first_split_point), std::real(second_split_point), 1e-16, 1e-9, workspace, &result_real, NULL, NULL);
				//gsl_integration_qawc()
			//gsl_integration_qags(&F_real, std::imag(first_split_point), std::imag(second_split_point), 1e-12, 0, n, workspace, &res_re2, &abs_error_real);

			gsl_function F_imag;
			F_imag.function = [](double x, void* p)->double {
				return std::imag((*static_cast<integrand_fun*>(p))(x));
			};
			F_imag.params = &integrand;
			//gsl_integration_qng(&F_imag, std::real(first_split_point), std::real(second_split_point), 1e-10, 1e-10, &result_imag, &abs_error_imag, &evals_imag);
			gsl_integration_qags(&F_imag, std::real(first_split_point), std::real(second_split_point), 1e-12, 1e-12, n, workspace, &result_imag, &abs_error_imag);
			//gsl_integration_qags(&F_imag, std::imag(first_split_point), std::imag(second_split_point), 1e-12, 0, n, workspace, &res_im2, &abs_error_imag);

			//auto status2 = gsl_integration_cquad(&F_imag, std::real(first_split_point), std::real(second_split_point), 1e-16, 1e-9, workspace, &result_imag, NULL, NULL);

			return std::complex<double>(result_real , result_imag );
		}
	private:
		mutable std::mutex integration_mutex;
		//gsl_integration_cquad_workspace* workspace;
		size_t n;
		gsl_integration_workspace* workspace;
	};
}