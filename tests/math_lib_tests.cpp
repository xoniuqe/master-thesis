#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include <steepest_descent/datatypes.h>
#include <steepest_descent/math_utils.h>
using namespace std::complex_literals;


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

TEST_CASE("splitpoints", "[math_lib]") {
	arma::mat  A{ {0, 0}, {2, 0}, {0, 2} };

	arma::vec b{ 0,-1,0 };

	arma::vec r{ -0.198422940262896, -0.150399880277165, 0 };

	arma::vec theta{ 1,0, 0 };

	auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);
	auto q = 0.;
	auto s = 0.;
	auto left_split = 0.;
	auto right_split = 1.;
	auto k = 1000;

	SECTION("Check type of singulerity") {
		auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
		auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

		if (std::abs(std::imag(sing_point)) < std::abs(std::imag(c))) {
			sing_point = std::real(sing_point);
		}

		if (std::abs(std::imag(spec_point)) < std::abs(std::imag(c))) {
			spec_point = std::real(spec_point);
		}
		auto is_spec = math_utils::is_singularity_in_layer(0.1, spec_point, left_split, right_split);

		auto is_sing = math_utils::is_singularity_in_layer(0.1, sing_point, left_split, right_split);
		REQUIRE(is_sing);
		REQUIRE(!is_spec);
	}
	SECTION("Check accuracy of split points") {
		auto [sp1, sp2] = math_utils::get_split_points_sing(q, k, s, {c, c_0}, left_split, right_split);

		REQUIRE(std::real(sp1) == Approx(0.408565556448243));
		REQUIRE(std::real(sp2) == Approx(0.441034563274592));

	}
}


TEST_CASE("get complex roots", "[math_lib]") {


	SECTION("get complex root of real number") {
		arma::mat  A{ {1, 1}, {1, 1,} , {0 ,0 } };

		arma::vec b{ 0,0,0 };

		arma::vec r{ 0, 1, 2 };
		auto [c, c_0] = math_utils::get_complex_roots(0, A, b, r);

		REQUIRE(std::real(c) == Approx(0.5));
		REQUIRE(std::imag(c) == Approx(1.5));
		REQUIRE(c_0 == Approx(2));
	}
	SECTION(" get compelx root of complex number") {

		arma::mat  A{ {1, 1}, {1, 1,} , {0 ,0 } };

		arma::vec b{ 0,0,0 };

		arma::vec r{ 0, 1, 2 };
		auto [c, c_0] = math_utils::get_complex_roots(1. + 1.i, A, b, r);

		REQUIRE(std::real(c) == Approx(-0.5));
		REQUIRE(std::imag(c) == Approx(0.5));
		REQUIRE(c_0 == Approx(2));
	}
	SECTION("Bug 1") {

		arma::mat  A{ {0, 0}, {0, 2}, {2, 0 } };

		arma::vec b{ 0,-0.5,0 };

		arma::vec r{ 0.0618, 0.1902, 0 };

		auto [c, c_0] = math_utils::get_complex_roots(0,  A, b, r);
		//0.346486400620223i
		REQUIRE(std::real(c) == Approx(0.));
		REQUIRE(std::imag(c) == Approx(0.34648));
		REQUIRE(c_0 == Approx(4));
	}

	SECTION(" Config 8") {


		arma::mat  A{ {1, 1}, {1, 1,} , {0 ,1 } };

		arma::vec b{ 1,0,-1 };

		arma::vec r{ 0, 1, 2 };

		auto [c, c_0] =  math_utils::get_complex_roots(0., A, b, r);

		REQUIRE(std::real(c) == Approx(0.));
		REQUIRE(std::imag(c) == Approx(2.3452));
		REQUIRE(c_0 == Approx(2));


		auto [c2, c2_0] = math_utils::get_complex_roots(0.00005, A, b, r);

		REQUIRE(std::real(c2) == Approx(-0.00005));
		REQUIRE(std::imag(c2) == Approx(2.3452));
		REQUIRE(c2_0 == Approx(2));
	}
}