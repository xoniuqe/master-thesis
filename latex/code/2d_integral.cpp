integration_result = tbb::parallel_reduce(tbb::blocked_range(0, number_of_steps, 1), 0. + 0.i, [&](tbb::blocked_range<int> range, std::complex<double> integral) 
{
	integrator::gsl_integrator_2d integrator;

	for (int i = range.begin(); i < range.end(); ++i)
	{
		auto y = config.y_resolution * (double)i;
		auto u = y + config.y_resolution * 0.5;
		auto [c, c_0] = math_utils::get_complex_roots(u, A, b, r);
		auto sing_point = math_utils::get_singularity_for_ODE(q, { c, c_0 });
		auto spec_point = math_utils::get_spec_point(q, { c, c_0 });

		auto s = arma::dot(A.col(0), theta) * u + prod;

		auto is_spec = math_utils::is_singularity_in_layer(config.tolerance, spec_point, 0, 1 - u);
		auto is_sing = math_utils::is_singularity_in_layer(config.tolerance, sing_point, 0, 1 - u);


		auto integration_y =  get_partial_integral(A1, b, r, 0., q1, sx1, c1, c1_0, y, y + config.y_resolution);
		auto integration_1_minus_y =  get_partial_integral(A2, b, r, 1., q2, sx2, c2, c2_0, y, y + config.y_resolution);

		//usage of steepest_descent_2d seems to be slower?
		std::tuple<std::complex<double>, std::complex<double>> split_points;
		auto path = path_utils::get_weighted_path_2d(0, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
		auto Iin = gauss_laguerre::calculate_integral_cauchy_tbb(path, nodes, weights);

		auto path2 = path_utils::get_weighted_path_2d(1 - u, u, A, b, r, q, config.wavenumber_k, s, { c, c_0 }, sing_point);
		auto Ifin = gauss_laguerre::calculate_integral_cauchy_tbb(path2, nodes, weights);

		integral += Iin * integration_y - Ifin * integration_1_minus_y;

		if (!is_spec && !is_sing) {
			// no singularity
			continue;
		}
		else if (is_spec) {
			split_points = math_utils::get_split_points_spec(q, config.wavenumber_k, s, { c, c_0 }, 0, 1 - u);
		}
		else if (is_sing) {
			split_points = math_utils::get_split_points_sing(q, config.wavenumber_k, s, { c, c_0 }, 0, 1 - u);
		}
		auto& [split_point1, split_point2] = split_points;

		//singularity
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
	}
	return integral;
}, std::plus<std::complex<double>>());