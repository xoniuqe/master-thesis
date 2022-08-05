#include <benchmark/benchmark.h>
#include <armadillo>
#include <steepest_descent/path_utils.h>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/integrator.h>
#include <steepest_descent/integral_1d.h>
#include <steepest_descent/integral_2d.h>
#include <vector>
#include <algorithm>


static void BM_integral2d(benchmark::State& state) {

/*resY = 0.5 * 10 ^ -3;
resY_std = 0.1;
% K = [5]; %;
K = [100, 500, 1000 3000, 5000]
r         = [1;0;1]; % Observer
mu        = [1;4;0];
A         = [0,0;2,0;0,2];
b         = [0;-0.5;0];

std::vector<double> K = { 100, 500, 1000, 3000, 5000 };*/
	for (auto _ : state) {
		arma::mat  A{ {0, 0}, {2, 0}, {0, 2 } };

		arma::vec3 b{ 0,-0.5,0 };

		arma::vec3 r{ 1., 0. ,1. };
		arma::vec3 mu{ 1., 4., 0. };

		std::vector<double> alphas;
		for (auto n = -0.5; n <= 0.5; n += 0.05) {
			alphas.push_back(n);
		}
		//std::generate(std::begin(alphas), std::end(alphas), [n = -0.5]() mutable { return n + 0.05; } );
		std::vector<double> radia;
		for (auto n = 0.2; n <= 3.8; n += 0.4) {
			radia.push_back(n);
		}

		config::configuration_2d config;
		config.wavenumber_k = 3000;
		config.tolerance = 0.1;
		config.y_resolution = 0.1;
		config.gauss_laguerre_nodes = 1000;

		integrator::gsl_integrator gslintegrator;
		integrator::gsl_integrator_2d gsl_integrator_2d;
		integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);
		for (auto radius : radia) {
			for (auto alpha : alphas) {
				arma::mat rotation{ {cos(alpha), -sin(alpha) , 0.}, {sin(alpha), cos(alpha), 0.}, { 0., 0., 1. } };
				auto local_r = radius * rotation * r;
				auto res = integral2d(A, b, r, mu);
			}

		}
	}
}

// Register the function as a benchmark
BENCHMARK(BM_integral2d);


BENCHMARK_MAIN();