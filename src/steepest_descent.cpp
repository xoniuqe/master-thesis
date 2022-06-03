#include <steepest_descent/steepest_descent.h>
#include <steepest_descent/gauss_laguerre.h>
#include <steepest_descent/path_utils.h>

namespace steepest_descent {

	steepest_descend_1d::steepest_descend_1d(std::vector<double> nodes, std::vector<double> weights, const double k, const double y, const datatypes::matrix& A, const  datatypes::vector& b, const  datatypes::vector& r, 
		const double q, const double s, const datatypes::complex_root complex_root, const std::complex<double> sing_point) 
		: nodes(nodes), weights(weights), k(k), y(y), A(A), b(b), r(r), q(q), s(s), complex_root(complex_root), sing_point(sing_point) {

	}

	auto steepest_descend_1d::operator()(const std::complex<double> first_split, const std::complex<double> second_split) const -> std::complex<double> {
		auto first_path = path_utils::get_weighted_path(first_split, y, A, b, r, q, k, s, complex_root, sing_point);
		auto second_path = path_utils::get_weighted_path(second_split, y, A, b, r, q, k, s, complex_root, sing_point);

		return gauss_laguerre::calculate_integral_cauchy(first_path, second_path, nodes, weights);
	}
}