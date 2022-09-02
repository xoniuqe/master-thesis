#pragma once

#ifndef STEEPEST_DESC_API
#define STEEPEST_DESC_API

#include "configuration.h"
#include "integral_2d.h"
#include "integrator.h"
#include "datatypes.h"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/blocked_range.h>

#include <complex>
#include <vector>
#include <armadillo>

namespace api {
	auto calculate_acoustic_single_layer(const config::configuration_2d& config, const std::vector<triangle> triangles, const std::vector<arma::vec3> observation_points, const arma::vec3 direction) -> std::complex<double> {
		integrator::gsl_integrator gslintegrator;
		integrator::gsl_integrator_2d gsl_integrator_2d;
		integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);


		auto integration_result = tbb::parallel_reduce(tbb::blocked_range(0, observation_points.count()), 0. + 0.i, [&](tbb::blocked_range<int> range, std::complex<double> integral) {
			for (int i = range.begin(); i < range.end(); ++i)
			{
				integral += tbb::parallel_reduce(tbb:blocked_range(0, triangles.count()), 0. + 0.i, [&integral2d, &triangle=triangles[i], &observation_points, &direction](tbb::blocked_range<int> range, std::complex<double> integral) {
					for (int i = range.begin(); i < range.end(); ++i)
					{
						integral += integral2d(triangle.A, triangle.b, observation_points[i], direction);
					}					);
			});
		return integration_result;

	}
}


#endif