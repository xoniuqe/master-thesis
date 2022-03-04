﻿// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include <gsl/gsl_linalg.h>


using namespace std;

int main()
{
	cout << "Hello CMake." << endl;

	double A_data[] = {
		  0.57092943, 0.00313503, 0.88069151, 0.39626474,
		 0.33336008, 0.01876333, 0.12228647, 0.40085702,
		  0.55534451, 0.54090141, 0.85848041, 0.62154911,
		 0.64111484, 0.8892682 , 0.58922332, 0.32858322
	};

	double b_data[] = {
		1.5426693 , 0.74961678, 2.21431998, 2.14989419
	};

	// Access the above C arrays through GSL views
	auto A = gsl_matrix_view_array(A_data, 4, 4);
	auto b = gsl_vector_view_array(b_data, 4);

	// Print the values of A and b using GSL print functions
	std::cout << "A = \n";
	gsl_matrix_fprintf(stdout, &A.matrix, "%lf");

	std::cout << "\nb = \n";
	gsl_vector_fprintf(stdout, &b.vector, "%lf");

	// Allocate memory for the solution vector x and the permutation perm:
	gsl_vector* x = gsl_vector_alloc(4);
	gsl_permutation* perm = gsl_permutation_alloc(4);

	// Decompose A into the LU form:
	int signum;
	gsl_linalg_LU_decomp(&A.matrix, perm, &signum);

	// Solve the linear system
	gsl_linalg_LU_solve(&A.matrix, perm, &b.vector, x);

	// Print the solution
	std::cout << "\nx = \n";
	gsl_vector_fprintf(stdout, x, "%lf");

	// Release the memory previously allocated for x and perm
	gsl_vector_free(x);
	gsl_permutation_free(perm);

	return 0;
}
