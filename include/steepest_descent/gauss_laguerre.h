#pragma once

#include <vector>
#include <tuple>
#include "path_utils.h"

namespace gauss_laguerre {



	auto calculate_laguerre_points_and_weights(size_t n) -> std::tuple<std::vector<double>, std::vector<double>>;
	
	auto calculate_integral_cauchy(const path_utils::path_function paths, std::vector<double> nodes, std::vector<double> weights)->std::complex<double>;


	auto calculate_integral_cauchy(const path_utils::path_function first_path, const path_utils::path_function second_path, std::vector<double> nodes, std::vector<double> weights)->std::complex<double>;
}