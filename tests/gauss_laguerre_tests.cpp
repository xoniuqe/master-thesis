#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include "../SteepestDescent/gauss_laguerre.h"


TEST_CASE("laguerre points", "[gauss_laguerre]") {
	auto [laguerre_points, x, y] = gauss_laguerre::calculate_laguerre_points_and_weights(10);

}