auto calculate_integral_cauchy(const path_utils::path_function first_path, const path_utils::path_function second_path, std::vector<double> nodes, std::vector<double> weights)->std::complex<double> 
{
    auto size = (int)nodes.size();
    // Parallelisierung
    auto result = tbb::parallel_deterministic_reduce(tbb::blocked_range(0, size, 100), 0. + 0.i, [&](tbb::blocked_range<int> range, std::complex<double> integral)
        {
            // Auswertung für den Block
            for (auto i = range.begin(); i < range.end(); ++i)
            {
                // Auswertung von $path$ an Laguerre-Konten i mit entsprechendem Gewicht
                integral += (weights[i] * path(nodes[i]));
            }
            return integral;
        }, std::plus<std::complex<double>>());
    return result;
}