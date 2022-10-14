#ifndef GAUSS_LAGUERRE_HEADER
#define GAUSS_LAGUERRE_HEADER

#pragma once


#include <vector>
#include <tuple>
#include "path_utils.h"
#include <numeric>
#include <algorithm> 

#ifdef _WIN32
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#else
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/blocked_range.h>
#endif

namespace gauss_laguerre {
	using namespace std::literals::complex_literals;



	auto calculate_laguerre_points_and_weights(size_t n) -> std::tuple<std::vector<double>, std::vector<double>>;
	
	//note: templated is slower than using path_function, inline and constexpr are nearly identical and static is way worse
#if  defined(_MSC_VER)
	constexpr auto calculate_integral_cauchy(const path_utils::path_function& path, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double> {
#else
	inline auto calculate_integral_cauchy(const path_utils::path_function & path, const std::vector<double>&nodes, const std::vector<double>&weights)->std::complex<double> {
#endif
		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), 0. + 0.i, std::plus<std::complex<double>>(), [&](const auto& left, const auto& right) -> auto {
			//slower: return right * (first_path(left) - second_path(left));
			return (right * path(left));
			});
	}

	//inline oder in cpp schieben!
	inline auto calculate_integral_cauchy_tbb(const path_utils::path_function& path, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double>
	{
		auto size = (int)nodes.size();
		auto result = tbb::parallel_deterministic_reduce(tbb::blocked_range(0, size, 100), 0. + 0.i, [&](tbb::blocked_range<int> range, std::complex<double> integral)
			{
				for (auto i = range.begin(); i < range.end(); ++i)
				{
					integral += (weights[i] * path(nodes[i]));
				}
				return integral;
			}, std::plus<std::complex<double>>());
		return result;
	}

	auto calculate_integral_cauchy(const path_utils::path_function first_path, const path_utils::path_function second_path, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double>;
	
	
	/*template<typename T>
	auto calculate_integral_cauchy_2(const T& first_path, const T& second_path, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double> {
		return std::transform_reduce(nodes.begin(), nodes.end(), weights.begin(), 0. + 0.i, std::plus<std::complex<double>>(), [&](const auto left, const auto right) -> auto {
			//slower: return right * (first_path(left) - second_path(left));
			return (right * first_path(left) - right * second_path(left));
			});
	}
	/*template<typename ...T>
	auto calculate_integral_cauchy_alt(T... &path_functions, const std::vector<double>& nodes, const std::vector<double>& weights)->std::complex<double>[] {


		std::vector<std::complex<double>> eval_points1, eval_points2, eval_points_summed;
		std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points1), first_path);
		std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points2), second_path);
		std::transform(eval_points1.begin(), eval_points1.end(), eval_points2.begin(), std::back_inserter(eval_points_summed), [](const auto left, const auto right) -> auto {
			return left - right;
			});
		return std::inner_product(weights.begin(), weights.end(), eval_points_summed.begin(), 0. + 0.i);
	}*/

}
#endif