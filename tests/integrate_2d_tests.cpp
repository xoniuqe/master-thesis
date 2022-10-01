#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include <armadillo>
#include <steepest_descent/configuration.h>
#include <steepest_descent/path_utils.h>
#include <steepest_descent/math_utils.h>
#include <steepest_descent/integral_2d.h>


TEST_CASE("config0", "[integral2d]") {
	
	config::configuration_2d config;
	config.wavenumber_k = 10;
	config.y_resolution = 0.1;
	config.gauss_laguerre_nodes = 30;
	integrator::gsl_integrator gslintegrator;
	integrator::gsl_integrator_2d gsl_integrator_2d;
	integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d);


	SECTION("integrate 2d config 0") {
		arma::mat A{ {0,0},{1,0},{1,1} };
		arma::vec3 b{ 0,0,0 };
		arma::vec3 r{ 1,-0.5,-0.5 };
		arma::vec3 theta{ 1,-0.5,-0.5 };

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(0.269855805676434));
		REQUIRE(std::imag(result) == Approx(+0.006289688201800));
	}

	SECTION("integrate 2d config 1") {
		arma::mat A{ {-1,1},{2,-1},{-0.5,0} };
		arma::vec3 b{ 0.9,-2,10 };
		arma::vec3 r{ 10,1,10 };
		arma::vec3 theta{ 0,0,1 };

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(0.005535158911812));
		REQUIRE(std::imag(result) == Approx(-0.011534995630686));
	}

	SECTION("integrate 2d config 2") {
		arma::mat A{ {1,0},{0,1},{0,0} };
		arma::vec3 b {0,0,0};
		arma::vec3 r {0.5,-1,10};
		arma::vec3 theta {0,0,1};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(0.043493417229190));
		REQUIRE(std::imag(result) == Approx(+0.018554200693989));
	}

	SECTION("integrate 2d config 3") {
		arma::mat A {{4,1},{0,0},{-0.1,45}};
		arma::vec3 b {0,-1,2};
		arma::vec3 r {-1,0,2};
		arma::vec3 theta {-0.2,-1,1};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(-0.000110871767547));
		REQUIRE(std::imag(result) == Approx(+0.000010844836399495));
	}

	SECTION("integrate 2d config 4") {
		arma::mat A {{1,0},{1,1},{0,1}};
		arma::vec3 b {1,0,2};
		arma::vec3 r {1.5,1,2};
		arma::vec3 theta {1,-1,0};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(0.134946712344323));
		REQUIRE(std::imag(result) == Approx(-0.305064875481159));
	}

	SECTION("integrate 2d config 5") {
		arma::mat A {{-1,1},{2,-1},{-0.5,0}};
		arma::vec3 b {0.9,-2,10};
		arma::vec3 r {10,1,10};
		arma::vec3 theta {0,0,1};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(0.005535158911812));
		REQUIRE(std::imag(result) == Approx(-0.011534995630686));
	}

	SECTION("integrate 2d config 6") {
		arma::mat A {{-1,0},{2,1},{-0.5,0}};
		arma::vec3 b {0.5,-8,0};
		arma::vec3 r {10,1,10};
		arma::vec3 theta {0,0,1};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(-0.013474698946195));
		REQUIRE(std::imag(result) == Approx(-0.004181926949958));
	}

	SECTION("integrate 2d config 7") {
		arma::mat A {{-1,1},{2,-1},{-0.5,0}};
		arma::vec3 b {0,-8,0};
		arma::vec3 r {10,1,10};
		arma::vec3 theta {0,0,1};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(-0.005673112721857));
		REQUIRE(std::imag(result) == Approx(+0.008484796703767));
	}

	SECTION("integrate 2d config 8") {
		arma::mat A {{1,1},{1,1},{0,1}};
		arma::vec3 b {1,0,-1};
		arma::vec3 r {0,1,2};
		arma::vec3 theta {1,0,0};

		auto result = integral2d(A, b, r, theta);
		REQUIRE(std::real(result) == Approx(-0.004211510384111));
		REQUIRE(std::imag(result) == Approx(+0.007729343560569));
	}
}
/*
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
}*/