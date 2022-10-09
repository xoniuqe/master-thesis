#include <benchmark/benchmark.h>

#include <numeric>
#include <random>
#include <armadillo>
#include <steepest_descent/gauss_laguerre.h>
#include <steepest_descent/integral_1d.h>



//template <class ...Args>
void Benchmark_1DIntegration_Varying_K(benchmark::State& state, int k) {
	//auto args_tuple = std::make_tuple(std::move(args)...);
	//auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(600);
	for (auto _ : state) {
		arma::mat  A{ {0, 0}, {1, 0}, {0, 1 } };

		arma::vec b{ 0,0,0 };
		config::configuration config;
		config.wavenumber_k = k;
		config.tolerance = 0.1;
		config.gauss_laguerre_nodes = 600;

		integrator::gsl_integrator gslintegrator;
		integral::integral_1d integral(config, &gslintegrator);
		std::vector<int64_t> timings(40);


		std::random_device rd;  // Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist(0, 1);
		for (auto i = 0; i < 40; i++) {

			arma::vec3 r{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };
			arma::vec3 mu{ 10. * dist(gen) + 0.5, 5. * dist(gen) - 3., 0. };

			auto start = std::chrono::steady_clock::now();
			benchmark::DoNotOptimize(integral(A, b, r, mu, 0, 0, 1));
			auto end = std::chrono::steady_clock::now();;
			timings[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}
		auto average = std::accumulate(timings.begin(), timings.end(), 0.) / 40.;
		state.counters["Single time"] = benchmark::Counter(average, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
	}
}
BENCHMARK_CAPTURE(Benchmark_1DIntegration_Varying_K, K_100, 100)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_1DIntegration_Varying_K, K_500, 500)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_1DIntegration_Varying_K, K_1000, 1000)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_1DIntegration_Varying_K, K_3000, 3000)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();
BENCHMARK_CAPTURE(Benchmark_1DIntegration_Varying_K, K_5000, 5000)->Unit(benchmark::kMillisecond)->MeasureProcessCPUTime()->UseRealTime();


