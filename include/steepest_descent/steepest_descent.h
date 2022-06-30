#pragma once

#include "datatypes.h"
#include <complex>

namespace steepest_descent {

	struct steepest_descend_1d {
		steepest_descend_1d(std::vector<double> nodes, std::vector<double> weights, const double k, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, 
			const double q, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point);

		auto operator()(const std::complex<double> first_split, const std::complex<double> second_split) const ->std::complex<double>;
	private:
		std::vector<double> nodes, weights;
		double k, y, q, s;
		datatypes::matrix A;
		datatypes::vector b, r;
		datatypes::complex_root complex_root; 
		std::complex<double> sing_point;
	};

	struct steepest_descend_2d {
		steepest_descend_2d(std::vector<double> nodes, std::vector<double> weights, const double k, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r,
			const double q, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point);

		auto operator()(const std::complex<double> first_split, const std::complex<double> second_split) const->std::complex<double>;
	private:
		std::vector<double> nodes, weights;
		double k, y, q, s;
		datatypes::matrix A;
		datatypes::vector b, r;
		datatypes::complex_root complex_root;
		std::complex<double> sing_point;
	};
}