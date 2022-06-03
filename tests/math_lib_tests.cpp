#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include <steepest_descent/datatypes.h>
#include <steepest_descent/math_utils.h>


TEST_CASE("calculate spec and sing points", "[math_lib]") {
	arma::mat  A{ {1, 1}, {1, 1,} , {0 ,0 } };
	arma::vec b{ 0,0,0 };
	arma::vec r{ 0, 1, 2 };
	arma::vec mu{ 1,4,0 };
	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);
	auto q = -8;

	SECTION("calculate the spec point") {
		auto spec_point = math_utils::get_spec_point(q, { c, c_0 });
		REQUIRE(std::real(spec_point) == Approx(0.7694));
	}

	SECTION("calculate the sing point") {
		auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
		REQUIRE(std::real(sing_point) == Approx(0.5));
		REQUIRE(std::imag(sing_point) == Approx(1.524));
	}
}


TEST_CASE("get complex roots", "[math_lib]") {
	arma::mat  A{ {1, 1}, {1, 1,} , {0 ,0 } };

	arma::vec b{ 0,0,0 };

	arma::vec r{ 0, 1, 2 };

	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

	REQUIRE(std::real(c) == Approx(0.5));
	REQUIRE(std::imag(c) == Approx(1.5));
	REQUIRE(c_0  == Approx(2));
}