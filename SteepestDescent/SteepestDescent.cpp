// SteepestDescent.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "SteepestDescent.h"
#include  <complex> //include complex to replace the gsl complex numbers

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_laguerre.h>

using namespace std;
/*
typedef int inputType;
typedef int ouputType;
typedef ouputType (*function)(const inputType);
/*
auto createI(const function f, const function g) {
	return[](const inputType k const inputType a, const inputType b) -> {
	//	gsl_sf_laguerre_n(160, )
	};
}*/


/// <summary>
/// Calculates the partial derivative in x direction.
/// </summary>
/// <param name="x">Barycentrical parameter x of the triangle </param>
/// <param name="y">Barycentrical parameter y of the triangle </param>
/// <param name="A">Jacobian matrix of the triangle </param>
/// <param name="b">Affine transformation Ax + b </param>
/// <param name="r">View vector </param>
/// <returns>The partial derivative in x direction </returns>
auto partial_derivative_P_x(const double x, const double y, const gsl_matrix* A, const gsl_vector* b, const gsl_vector* r) {
	auto calculate_row = [&](const int i) {
		auto first_val = gsl_matrix_get(A, i, 0);
		auto second_val = gsl_matrix_get(A, i,1);
		auto b_element = gsl_vector_get(b, i);
		auto r_element = gsl_vector_get(r, i);
		return first_val * 2 * (first_val * x + second_val * y + b_element - r_element);
	};
	return calculate_row(0) + calculate_row(1) + calculate_row(2);
}

auto calculate_P_x(const gsl_vector* x, const gsl_matrix* A, const gsl_vector* b, const gsl_vector* r) {
	//gsl_matrix_mul
	//todo: implement this with matrix-vector-multiplication => the papers implementation seems to make things more complicated than needed
}

auto calculate_P_x(const double x, const double y, const gsl_matrix* A, const gsl_vector* b, const gsl_vector* r) {
	//fx = ((A(1, 1) * x + A(1, 2) * y + b(1) - r(1)). ^ 2 + (A(2, 1) * x + A(2, 2) * y + b(2) - r(2)). ^ 2 + (A(3, 1) * x + A(3, 2) * y + b(3) - r(3)). ^ 2);
	auto calculate_row = [&](const int i) {
		auto first_val = gsl_matrix_get(A, i, 0);
		auto second_val = gsl_matrix_get(A, i, 1);
		auto b_element = gsl_vector_get(b, i);
		auto r_element = gsl_vector_get(r, i);
		return gsl_pow_2((first_val * x + second_val * y + b_element - r_element));
	};
	return calculate_row(0) + calculate_row(1) + calculate_row(2);
}

auto dot_product(const gsl_vector_view* a, const gsl_vector* b) -> double;
auto dot_product(const gsl_vector_view a, const gsl_vector_view b) -> double;
auto dot_product(const gsl_vector_view * a, const gsl_vector_view * b) -> double;
auto dot_product(const gsl_vector * a, const gsl_vector * b) -> double;


auto dot_product(const gsl_vector_view* a, const gsl_vector* b) -> double {
	return dot_product(&a->vector, b);
}

auto dot_product(const gsl_vector_view a, const gsl_vector_view b) -> double {
	return dot_product(&a.vector, &b.vector);
}

auto dot_product(const gsl_vector_view* a, const gsl_vector_view* b) -> double {
	return dot_product(&a->vector, &b->vector);
}

auto dot_product(const gsl_vector* a, const gsl_vector* b) -> double {
	double result = 0;
	gsl_blas_ddot(a, b, &result);
	return result;
}

auto get_complex_roots(const double y, gsl_matrix* A, const gsl_vector* b, const gsl_vector* r) -> auto{
	auto get_column = [&](const int i) {
		return gsl_matrix_column(A, i);
	};
	auto A_1 = get_column(0);
	auto A_2 = get_column(1);

	auto A_2_copy = gsl_vector_alloc(3);
	gsl_vector_memcpy(&A_2.vector, A_2_copy);
	gsl_vector_scale(A_2_copy, y);
	gsl_vector_add(A_2_copy, b);
	gsl_vector_sub(A_2_copy, r);
	auto real_c = - dot_product(&A_1, A_2_copy) / dot_product(A_1, A_1);
	auto P_rc = calculate_P_x(real_c, y, A, b, r);

	double c_0;
	if (abs(real_c) > 0.000000001)
		c_0 = (calculate_P_x(0, y, A, b, r) - P_rc) / gsl_pow_2(real_c);
	else {
		auto x = calculate_P_x(1, y, A, b, r);
		c_0 = 1.0 / 2.0 * x;

	}
	

	auto c = real_c + 1i * sqrt(P_rc / c_0);
	return std::make_tuple(c, c_0);
}

auto stub_integrate_1d() {

}

void setup_1d_test()
{
	double A_data[]{ -5, 1, -1, 1, -1 ,0 };

	double b_data[] = { 0,1,0 };
	auto b = gsl_vector_view_array(b_data, 3);

	double r_data[] = { 0, 12, 1 };
	auto r = gsl_vector_view_array(r_data, 3);
	
	double mu_data[] = { 1,4,0 };
	//auto mu = gsl_vector_view_array(mu_data, 3);

	auto A = gsl_matrix_alloc(3, 2);
	for (auto i = 0; i < 3; i++) {
		for (auto j = 0; j < 2; j++) {
			gsl_matrix_set(A, i, j, A_data[(i * 2) + j]);
		}
	}
	auto DPx = partial_derivative_P_x(0, 0, A, &b.vector, &r.vector);

	std::cout << "\nDPx = \n" << DPx;

	auto [c, c_0] = get_complex_roots(0, A, &b.vector, &r.vector);

	std::cout << "\nc = \n" << c;
	std::cout << "\nc0 = \n" << c_0;



	/*auto mu = gsl_vector_alloc(3);
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
	double q = 0, s, s_1, s_2;
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
	
	delete A, mu, A_1, A_2;
	//auto s = (mu.vector[0] * A.matrix[0, 1] + mu.vector[1] * A.matrix[1, 1] + mu.vector[2] * A.matrix[2, 1]) * y+ dotgsl*/
}

constexpr int test()
{
	return 5;
}

int main()
{
#ifdef HAVE_INLINE
	cout << " HAVE INLINE " << endl;
#endif
	cout << "Hello CMake." << endl;
/*	double A_data[] = {
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
	*/

	setup_1d_test();


	return 0;
}

