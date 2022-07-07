#pragma once

#include <functional>
#include <complex>

#include "gsl_function_wrapper.h"

#include <gsl/gsl_complex.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_laguerre.h>



namespace integrator {
	using namespace std::complex_literals;


	//typedef std::function <const std::complex<double>(const std::complex<double>)> integrand_fun;

	struct gsl_integrator {
		gsl_integrator() {
			auto limit = 1000;
			workspace = gsl_integration_cquad_workspace_alloc(limit);
		}

		~gsl_integrator() {
			gsl_integration_cquad_workspace_free(workspace);
		}

		template<class integrand_fun>
		auto operator()(integrand_fun integrand, const std::complex<double> first_split_point, const std::complex<double> second_split_point) const ->std::complex<double>
		{
			return 0;
			/*
			double result_real, result_imag;
			//auto integrand_C_pointer = const_cast<void*>(integrand);
		
			gsl_function F_real;
			F_real.function = [](double x, void* p)->double {
				return std::real((*static_cast<integrand_fun*>(p))(x));
			};
			F_real.params = &integrand;

			gsl_function F_imag;
			F_imag.function = [](double x, void* p)->double {
				return std::imag((*static_cast<integrand_fun*>(p))(x));
			};
			F_imag.params = &integrand;
			//auto real_fun = wrap_real_part(integrand);
	

			gsl_integration_cquad(&F_real, std::real(first_split_point), std::real(second_split_point), 1e-10, 1e-6, workspace, &result_real, NULL, NULL);

			gsl_integration_cquad(&F_imag, std::real(first_split_point), std::real(second_split_point), 1e-10, 1e-6, workspace, &result_imag, NULL, NULL);

			return result_real + result_imag * 1i;*/
		}
	private:
		gsl_integration_cquad_workspace* workspace;
	};
}