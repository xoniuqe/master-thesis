#pragma once

#include <mutex>
#include <semaphore>
#include <vector>
#include <integration/gsl_integrator_2d>

namespace utils {
	template<int maximum_size>
	class integrator_pool_2d {
	public:

		template<class integrand_fun>
		auto calcualte_integral(integrand_fun integrand, const std::complex<double> x_start, const std::complex<double> x_end, const double y_start, const double y_end) const->std::complex<double>
		{
			sem_resource.aquire();

		}


	private:

		std::counting_semaphore<maximum_size> sem_resource;
		std::vector<gsl_integrator_2d*> resources;
	}
}