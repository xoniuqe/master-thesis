#pragma once
#include <armadillo>
#include <complex>

namespace datatypes {
	typedef  arma::mat matrix;


#define DOT_PRODUCT(v1, v2) arma::dot(v1,v2);


	struct triangle {
		matrix A;
		arma::vec3 b;
	};

	struct complex_root {
		std::complex<double> c;
		double c_0;
	};

}
