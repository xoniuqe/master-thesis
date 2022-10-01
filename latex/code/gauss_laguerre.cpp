auto calculate_integral_cauchy(const path_utils::path_function first_path, const path_utils::path_function second_path, std::vector<double> nodes, std::vector<double> weights)->std::complex<double> 
{
    std::vector<std::complex<double>> eval_points1, eval_points2, eval_points_summed;
    // Evaluierung entlang der Pfade
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points1), first_path);
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(eval_points2), second_path);
    // Summierung
    std::transform(eval_points1.begin(), eval_points1.end(), eval_points2.begin(), std::back_inserter(eval_points_summed), [](const auto left, const auto right) -> auto {
        return left - right;
        });
    // Gewichtung
    return std::inner_product(weights.begin(), weights.end(), eval_points_summed.begin(), 0. + 0.i);
}