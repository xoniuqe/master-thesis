#pragma once
#include <vector>
#include <algorithm>
#include <numeric>

#include <complex>
#include <armadillo>




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
	}



}