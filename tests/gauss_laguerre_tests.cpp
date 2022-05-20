#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include <steepest_descent/gauss_laguerre.h>



TEST_CASE("laguerre points", "[gauss_laguerre]") {
	std::vector<double> expected_laguerre_points = { 0.1378,0.7295,1.8083,3.4014,5.5525,8.3302,11.8438,16.2793,21.9966,29.9207};
	std::vector<double> expected_weights = {0.30844,0.40112,0.21807,0.062087,0.0095015,0.00075301,2.8259e-05,4.2493e-07,1.8396e-09,9.9118e-13};
	auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(10);

	for (auto i = 0; i < 10; i++) {
		REQUIRE(nodes[i] == Approx(expected_laguerre_points[i]).margin(0.0001));
		REQUIRE(weights[i] == Approx(expected_weights[i]).margin(0.0001));
	}
	

}