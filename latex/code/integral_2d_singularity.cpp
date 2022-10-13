// [...] 
if (!is_spec && !is_sing) {
    // Keine Singularität, Fall 1
    continue;
}
else if (is_spec) {
    split_points = math_utils::get_split_points_spec(q, config.wavenumber_k, s, { c, c_0 }, 0, 1 - u);
}
else if (is_sing) {
    split_points = math_utils::get_split_points_sing(q, config.wavenumber_k, s, { c, c_0 }, 0, 1 - u);
}
auto& [split_point1, split_point2] = split_points;

// Singularität entlang der Schicht 

auto path3 = path_utils::get_weighted_path_2d(split_point1, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
auto Ifin1 = gauss_laguerre::calculate_integral_cauchy_tbb(path3, nodes, weights);

auto path4 = path_utils::get_weighted_path_2d(split_point2, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
auto Iin2 = gauss_laguerre::calculate_integral_cauchy_tbb(path4, nodes, weights);


auto sx_intern_1 = arma::dot(A1.col(1), theta) * split_point1 + prod;
auto [cIntern1, c_0Intern1] = math_utils::get_complex_roots(split_point1, A1, b, r);
auto sx_intern_2 = arma::dot(A1.col(1), theta) * split_point2 + prod;
auto [cIntern2, c_0Intern2] = math_utils::get_complex_roots(split_point2, A1, b, r);

auto intYintern1 = get_partial_integral(A1, b, r, split_point1, q1, sx_intern_1, cIntern1, c_0Intern1, y, y + config.y_resolution);
auto intYintern2 = get_partial_integral(A1, b, r, split_point2, q1, sx_intern_2, cIntern2, c_0Intern2, y, y + config.y_resolution);


auto integral2_res = integrator.operator()(green_fun_2d, split_point1, split_point2, y, y + config.y_resolution);

auto integration_result = integral2_res + Iin2 * intYintern2  - Ifin1 * intYintern1;
integral += integration_result;