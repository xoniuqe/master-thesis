#pragma once
#include <gsl/gsl_complex.h>

namespace print_utils
{
	void print_matrix_pretty(gsl_matrix_complex* matrix, int size1, int size2) {
		for (int i = 0; i < size1; i++) {
			for (int j = 0; j < size2; j++) {
				auto value = gsl_matrix_complex_get(matrix, i, j);
				std::cout << GSL_REAL(value) << " ";// +" << GSL_IMAG(value) << "i ";
			}
			std::cout << std::endl;
		}
	}

	void print_matrix_pretty(gsl_matrix* matrix, int size1, int size2) {
		for (int i = 0; i < size1; i++) {
			for (int j = 0; j < size2; j++) {
				auto value = gsl_matrix_get(matrix, i, j);
				std::cout << value << " ";
			}
			std::cout << std::endl;
		}
	}
}