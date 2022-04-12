#pragma once
#include <vector>
#include <algorithm>
#include <numeric>

#include <complex>
#include <armadillo>

//#include <complex>
//#include <armadillo>




namespace math_utils {

	

	auto get_laguerre_points(const int n) {

	}

	auto lagpts(int n) {
		/*alpha = 2 * (1:n) - 1;  beta = 1:(n - 1); % 3 - term recurrence coeffs
			T = diag(beta, 1) + diag(alpha) + diag(beta, -1);% Jacobi matrix
			[V, D] = eig(T);% eigenvalue decomposition
			[x, indx] = sort(diag(D));% Laguerre points
			w = V(1, indx). ^ 2;% Quadrature weights
			v = sqrt(x).*abs(V(1, indx)).';        % Barycentric weights
			v = v. / max(v); v(2:2 : n) = -v(2:2 : n);*/
		std::vector<double> alpha(n);
		std::iota(std::begin(alpha), std::end(alpha), 1);
		std::for_each(std::begin(alpha), std::end(alpha), [](auto& x) { x = 2. * x - 1.; });
		std::vector<double> beta(n-1);
		
		std::iota(std::begin(beta), std::end(beta), 0);
		arma::mat T = arma::diagmat(arma::vec(beta), 1) + arma::diagmat(arma::vec(alpha)) + arma::diagmat(arma::vec(beta), -1);
		arma::cx_vec eigval;
		arma::cx_mat eigvec;
		auto eigen = arma::eig_gen(eigval, eigvec, T);
		//auto diag = arma::diagvec(eigvec);

		auto F = std::sqrt(C * C - (q * q / complex_root.c_0 * cTimesCconj - C * C) / (q * q / complex_root.c_0 - 1));

		return q < 0 ? C + F : C - F;
	}

	auto get_spec_point(const double q, const datatypes::complex_root complex_root) {
		auto C = std::real(complex_root.c);
		auto c_real_squared = std::pow(C, 2);
		auto rc = std::real(complex_root.c);
		auto ic = std::imag(complex_root.c);

		auto cTimesCconj = rc * rc + ic * ic;
		auto K = std::sqrt(1. / (complex_root.c_0 - q * q) * (q * q * C * C - complex_root.c_0 * cTimesCconj) + C * C);

		return q < 0 ? C + K : C - K;
	}
}