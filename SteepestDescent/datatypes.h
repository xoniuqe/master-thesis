#pragma once
#ifdef USE_ARMADILLO
#include <armadillo>
#endif
#include <complex>

namespace datatypes {
#ifdef USE_ARMADILLO
	typedef  arma::mat matrix;
	typedef  arma::vec vector;

#define VEC(name, size) arma::vec name(size);
#define MATRIX_GET(m,x,y) m.at(x,y)
#define VECTOR_GET(v,x) v.at(x);

#define DOT_PRODUCT(v1, v2) arma::dot(v1,v2);
#endif


	struct triangle {
		matrix* A;
		vector* b;
	};

	struct complex_root {
		std::complex<double> c;
		double c_0;
	};

}