// Berechnung der Schrittparameter 
auto y = config.y_resolution * (double)i;
auto u = y + config.y_resolution * 0.5;

//Berechnung der potentielllen Singularitäten
auto [c, c_0] = math_utils::get_complex_roots(u, A, b, r);
auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
auto spec_point = math_utils::get_spec_point(q, { c, c_0 });


// Prüfen ob Singulariäten vorliegen
auto is_spec = math_utils::is_singularity_in_layer(config.tolerance, spec_point, 0, 1 - u);
auto is_sing = math_utils::is_singularity_in_layer(config.tolerance, sing_point, 0, 1 - u);

// Berechnung der $\Lambda$-Terme mithilfe des 1D-Verfahrens			
auto integration_y = partial_integral(A1, b, r, q1, sx1, 0., c1, c1_0, y, y + config.y_resolution);
auto integration_1_minus_y = partial_integral(A2, b, r, q2, sx2, 1., c2, c2_0, y, y + config.y_resolution);

// Berechnung der Pfade mithilfe der Gauss-Laguerre-Quadratur 
auto s = arma::dot(A.col(0), theta) * u + prod;
auto path = path_utils::get_weighted_path_2d(0, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
auto Iin = gauss_laguerre::calculate_integral_cauchy_tbb(path, nodes, weights);

auto path2 = path_utils::get_weighted_path_2d(1 - u, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
auto Ifin = gauss_laguerre::calculate_integral_cauchy_tbb(path2, nodes, weights);

//Ergebniss für den Layer
integral += Iin * integration_y - Ifin * integration_1_minus_y;
//[...] Abhandlung der Singularitäten