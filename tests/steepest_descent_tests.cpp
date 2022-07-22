#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include <steepest_descent/math_utils.h>
#include <steepest_descent/gauss_laguerre.h>
#include <steepest_descent/steepest_descent.h>

TEST_CASE("templated func and direct impl match", "[steepest_descent]") {
	arma::mat A{ {0, 0}, {1, 0}, {1, 1 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 1, -0.5, -0.5 };

	arma::vec mu{ 1,-0.5, -0.5 };
	auto y = 0.;
	auto k = 100;
	auto left_split = 0.;
	auto right_split = 1.;

	auto [nodes, weights] = gauss_laguerre::calculate_laguerre_points_and_weights(30);

	auto q = arma::dot(A.col(0), mu);
	auto s = arma::dot(A.col(1), mu) * y + arma::dot(mu, b);
	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

	auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
	auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

	if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) {
		sing_point = std::real(sing_point);
	}

	if (std::abs(std::imag(spec_point) < std::abs(std::imag(c)))) {
		spec_point = std::real(spec_point);
	}

	auto [sp1, sp2] = math_utils::get_split_points_sing(q, k, s, { c, c_0 }, left_split, right_split);

	steepest_descent::steepest_descend_1d direct_impl(nodes, weights, k, y, A, b, r, q, s, { c, c_0 }, sing_point);

	steepest_descent::steepest_descend_2d templated_impl(path_utils::get_weighted_path, nodes, weights, k, y, A, b, r, q, s, { c, c_0 }, sing_point);

	auto result1_direct = direct_impl(left_split, sp1);
	auto result1_templated = templated_impl(left_split) - templated_impl(sp1);

	auto result2_direct = direct_impl(sp2, right_split);
	auto result2_templated = templated_impl(sp2) - templated_impl(right_split);

	REQUIRE(std::real(result1_direct) == Approx(std::real(result1_templated)).margin(0.0001));
	REQUIRE(std::imag(result1_direct) == Approx(std::imag(result1_templated)).margin(0.0001));
	REQUIRE(std::real(result2_direct) == Approx(std::real(result2_templated)).margin(0.0001));
	REQUIRE(std::imag(result2_direct) == Approx(std::imag(result2_templated)).margin(0.0001));
}