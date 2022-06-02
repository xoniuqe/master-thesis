#pragma once

#include <vector>
#include <tuple>
#include "path_utils.h"

namespace gauss_laguerre {



	auto calculate_laguerre_points_and_weights(int n) -> std::tuple<std::vector<double>, std::vector<double>>;
	
	auto calculate_integral(const path_utils::path_function path, std::vector<double> nodes, std::vector<double> weights) -> std::complex<double>;

	auto calculate_integral(const path_utils::path_function path1, const path_utils::path_function path2, std::vector<double> nodes, std::vector<double> weights)->std::complex<double>;
}