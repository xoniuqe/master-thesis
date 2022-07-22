#include <catch2/catch.hpp>
#include <complex>
#include <armadillo>

#include <steepest_descent/integration/gsl_integrator.h>
#include <steepest_descent/integration/gsl_integrator_2d.h>

#include <steepest_descent/math_utils.h>

using namespace std::complex_literals;

TEST_CASE("1d integration", "[integration]") {
	arma::mat  A{ {0, 0}, {1, 0} , {1 ,1 } };
	arma::vec b{ 0,0,0 };
	arma::vec r{ 1, -0.5, -0.5 };
	arma::vec mu{ 1, -0.5, -0.5 };

	auto y = 0;
	auto k = 100.;

	auto q = arma::dot(A.col(0), mu);
	auto s = arma::dot(A.col(1), mu) * y + arma::dot(mu, b);


	auto green_fun = [&](const double x) -> auto {
		auto Px = math_utils::calculate_P_x(x, y, A, b, r);
		auto sqrtPx = std::sqrt(Px);
		auto res = std::exp(1.i * k * (sqrtPx + q * x + s)) * (1. / sqrtPx);
		return res;
	};
	integrator::gsl_integrator integrator;
	auto x = integrator(green_fun, 0, 1);

	REQUIRE(std::real(x) == Approx(-0.090083).margin(0.00001));
	REQUIRE(std::imag(x) == Approx(0.232279).margin(0.00001));
}

/* integral2_res = integral2(greenFun2D,real(splitPt1),real(splitPt2),y,y+resY);
>> integral2_res
integral2_res =  3.8881e-05 + 1.5122e-05i


*/

TEST_CASE("2d integration identity", "[integration]") {
	auto identity = [](const double x, const double y) -> auto { return 1.; };
	integrator::gsl_integrator_2d integrator;
	auto x = integrator(identity, 0, 1, 0, 1);
	REQUIRE(std::real(x) == Approx(1));
	REQUIRE(std::imag(x) == Approx(0));
}

TEST_CASE("2d integration complex result", "[integration]") {
	auto complex1 = [](const double x, const double y) -> auto { return std::complex<double>(x, y); };
	integrator::gsl_integrator_2d integrator;
	auto x = integrator(complex1, 0, 1, 0, 1);
	REQUIRE(std::real(x) == Approx(.5));
	REQUIRE(std::imag(x) == Approx(.5));

	auto complex2 = [](const double x, const double y) -> auto { return std::complex<double>(x, -y); };
	x = integrator(complex2, 0, 1, 0, 1);
	REQUIRE(std::real(x) == Approx(.5));
	REQUIRE(std::imag(x) == Approx(-.5));
}

TEST_CASE("2d integration green_fun", "[integration]") {
	arma::mat  A{ {0, 0}, {1, 0} , {1 ,1 } };
	arma::vec b{ 0,0,0 };
	arma::vec r{ 1, -0.5, -0.5 };
	arma::vec mu{ 1, -0.5, -0.5 };
	auto k = 10.;

	auto qx = arma::dot(A.col(0), mu);
	auto qy = arma::dot(A.col(1), mu);
	auto prod = arma::dot(mu, b);

	REQUIRE(qx == Approx(-1));
	REQUIRE(qy == Approx(-0.5));
	REQUIRE(prod == Approx(0.));


	//>>     greenFun2D = @(x,y) exp(1i*k*(Px(x,y,A,b,r).^(1/2)+qx*x+qy*y+prod)).*Px(x,y,A,b,r).^(-1/2);
	auto green_fun_2d = [&k, &A, &b, &r, qx, qy, prod](const double x, const double y) -> auto {
		auto Px = math_utils::calculate_P_x(x, y, A, b, r);
		auto sqrtPx = std::sqrt(Px);
		auto res = std::exp(1.i * k * (sqrtPx + qx * x + qy * y + prod)) * (1. / sqrtPx);
		return res;
	};
	auto gf1 = green_fun_2d(0, 0); //0.7753239246961696 - 0.2560067937778117i
	REQUIRE(std::real(gf1) == Approx(0.7753239246961696));
	REQUIRE(std::imag(gf1) == Approx(-0.2560067937778117));
	auto gf2 = green_fun_2d(0, 1); //0.2222825145004821 + 0.4861118898583213i
	REQUIRE(std::real(gf2) == Approx(0.2222825145004821));
	REQUIRE(std::imag(gf2) == Approx(0.4861118898583213));
	auto gf3 = green_fun_2d(1, 0); //0.2698017831812562 + 0.3301896116027823i
	REQUIRE(std::real(gf3) == Approx(0.2698017831812562));
	REQUIRE(std::imag(gf3) == Approx(0.3301896116027823));
	auto gf4 = green_fun_2d(1, 1); //-0.3223329534136753 - 0.03694083158720506i
	REQUIRE(std::real(gf4) == Approx(-0.3223329534136753));
	REQUIRE(std::imag(gf4) == Approx(-0.03694083158720506));

	integrator::gsl_integrator_2d integrator;
	auto x = integrator(green_fun_2d, 0, 1, 0, 1);
	REQUIRE(std::real(x) == Approx(0.3039434217462966));
	REQUIRE(std::imag(x) == Approx(0.1840843367985873));
}
