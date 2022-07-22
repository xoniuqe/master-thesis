#pragma once

#include "math_utils.h"
#include <armadillo>
#include <complex>

namespace path_utils {

	using namespace std::literals::complex_literals;
	typedef std::function <const std::complex<double>(const double t)> path_function;


	auto generate_K_x(const std::complex<double> x, const std::complex<double> P_x, const double q)->std::function<const std::complex<double>(const double t)>;

	/*
	Idea: split this function up into path calculation and derivative. Motivation: we should construct this in a way to minimize nested lambdas to inline as much as possible!
	*/
	auto get_complex_path(const std::complex<double> split_point, const double y, const arma::mat& A, const arma::vec3& b, const  arma::vec3& r, const double q, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->std::tuple<path_function, path_function>;
	auto get_complex_path(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const arma::vec3& b, const  arma::vec3& r, const double q, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->std::tuple<path_function, path_function>;
	
	auto get_weighted_path_1d(const std::complex<double> split_point, const double y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	auto get_weighted_path(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point) ->path_function;
	auto get_weighted_path_2d(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	//auto get_weighted_path_2d(const std::complex<double> split_point, const double y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
	auto get_weighted_path_y(const std::complex<double> split_point, const std::complex<double> y, const arma::mat& A, const  arma::vec3& b, const  arma::vec3& r, const double q, const double k, const std::complex<double> s, const datatypes::complex_root complex_root, const std::complex<double> sing_point)->path_function;
}