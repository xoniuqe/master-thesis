#include <benchmark/benchmark.h>

#include <functional>
#include <numeric>
#include <random>
#include <armadillo>
#include <steepest_descent/gauss_laguerre.h>
#include <steepest_descent/integral_2d.h>



void Benchmark_2DIntegration_Varying_K(benchmark::State& state, int k) {
    auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(600);
    for (auto _ : state) {
		arma::mat  A{ {0, 0}, {1, 0}, {0, 1 } };

		arma::vec b{ 0,0,0 };
		config::configuration_2d config;
		config.wavenumber_k = k;
		config.tolerance = 0.1;
		config.y_resolution = 0.1;
		config.gauss_laguerre_nodes = 600;

		integrator::gsl_integrator gslintegrator;
		integrator::gsl_integrator_2d gsl_integrator_2d;
		integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d, nodes, weights);
		std::vector<int64_t> timings(40);


		std::random_device rd;  // Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist(0, 1);
		for (auto i = 0; i < 40; i++) {

			arma::vec3 r{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };
			arma::vec3 mu{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };

			auto start = std::chrono::steady_clock::now();			
			benchmark::DoNotOptimize(integral2d(A, b, r, mu));
			auto end = std::chrono::steady_clock::now();;
			timings[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}
		auto average = std::accumulate(timings.begin(), timings.end(), 0.) / 40.;
		state.counters["Single time"] = benchmark::Counter(average, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
    }
}
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_K, K_100, 100)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_K, K_500, 500)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_K, K_1000, 1000)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_K, K_3000, 3000)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_K, K_5000, 5000)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();


void Benchmark_2DIntegration_Varying_resolution(benchmark::State& state, double resolution) {
	auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(600);
	for (auto _ : state) {
		arma::mat  A{ {0, 0}, {1, 0}, {0, 1 } };

		arma::vec b{ 0,0,0 };
		config::configuration_2d config;
		config.wavenumber_k = 1000;
		config.tolerance = 0.1;
		config.y_resolution = resolution;
		config.gauss_laguerre_nodes = 600;

		integrator::gsl_integrator gslintegrator;
		integrator::gsl_integrator_2d gsl_integrator_2d;
		integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d, nodes, weights);
		std::vector<int64_t> timings(40);


		std::random_device rd;  // Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist(0, 1);
		for (auto i = 0; i < 40; i++) {

			arma::vec3 r{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };
			arma::vec3 mu{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };

			auto start = std::chrono::steady_clock::now();
			benchmark::DoNotOptimize(integral2d(A, b, r, mu));
			auto end = std::chrono::steady_clock::now();;
			timings[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}
		auto average = std::accumulate(timings.begin(), timings.end(), 0.) / 40.;
		auto std = std::sqrt(std::transform_reduce(timings.begin(), timings.end(), 0., std::plus<double>(), [&](auto& timing) {
			return (timing - average) * (timing - average);
			}) / 40.);
		state.counters["bla"] += 1;
		state.counters["Single time"] = average * 1000.;// benchmark::Counter(average, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
		state.counters["time std"] = std * 1000.;// benchmark::Counter(std, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
	}
}
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_resolution, RES_0_1, 0.1)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_resolution, RES_0_01, 0.01)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_2DIntegration_Varying_resolution, RES_0_001, 0.001)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
