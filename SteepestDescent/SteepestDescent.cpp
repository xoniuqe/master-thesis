// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include  <complex> //include complex to replace the gsl complex numbers

#include <gsl/gsl_linalg.h>


using namespace std;
void setup_1d_test()
{
	double A_data[]{ -5, 1, -1, 1, -1 ,0 };

	double b_data[] = { 0,1,0 };
	auto b = gsl_vector_view_array(b_data, 3);

	double mu_data[] = { 1,4,0 };
	//auto mu = gsl_vector_view_array(mu_data, 3);

	auto A = gsl_matrix_alloc(2, 3);
	for (auto i = 0; i < 2; i++) {
		for (auto j = 0; j < 3; j++) {
			gsl_matrix_set(A, i, j, A_data[i + (j * 2)]);
		}
	}
	auto mu = gsl_vector_alloc(3);
	for (auto j = 0; j < 3; j++) {
		gsl_vector_set(mu, j, mu_data[j]);
	}
	auto A_1 = gsl_vector_alloc(3);
	auto A_2 = gsl_vector_alloc(3);
	for (auto j = 0; j < 3; j++) {
		gsl_vector_set(A_1, j, A_data[j * 2]);
		gsl_vector_set(A_2, j, A_data[1 + (j * 2)]);
	}
	std::cout << "A = \n";
	gsl_matrix_fprintf(stdout, A, "%lf");
	std::cout << "A(1) = \n";
	gsl_vector_fprintf(stdout, A_1, "%lf");
	std::cout << "A(2) = \n";
	gsl_vector_fprintf(stdout, A_2, "%lf");

	auto y = 0;

	auto a_column_1 = gsl_matrix_column(A, 0);
	auto a_column_2 = gsl_matrix_column(A, 1);
	double q, s, s_1, s_2;
	//gsl_blas_sdot
	gsl_blas_ddot(mu, A_1, &q);
	gsl_blas_ddot(mu, A_2, &s_1);

	//gsl_blas_ddot(&mu.vector, &a_column_1.vector, &q);
	//gsl_blas_ddot(&mu.vector, &a_column_2.vector, &s_1);
	s_1 *= y;
	gsl_blas_ddot(mu, &b.vector, &s_2);
	//gsl_blas_ddot(&mu.vector, &b.vector, &s_2);
	s = s_1 + s_2;

	std::cout << "\nq = \n" << q;
	std::cout << "\ns = \n" << s;

	//auto s = (mu.vector[0] * A.matrix[0, 1] + mu.vector[1] * A.matrix[1, 1] + mu.vector[2] * A.matrix[2, 1]) * y+ dotgsl
}


int main()
{
#ifdef HAVE_INLINE
	cout << " HAVE INLINE " << endl;
#endif
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


	setup_1d_test();


	return 0;
}

