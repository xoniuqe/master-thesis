#pragma once

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <complex>

namespace datatypes {

	struct triangle {
		gsl_matrix* A;
		gsl_vector* b;
	};

	struct complex_root {
		gsl_complex c;// std::complex<double> c;
		double c_0;
	};

}