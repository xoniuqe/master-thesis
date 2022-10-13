integration_result = tbb::parallel_reduce(tbb::blocked_range(0, number_of_steps, 1), 0. + 0.i, [&](tbb::blocked_range<int> range, std::complex<double> integral) 
{
    // [...] Berechnung des 2D-Integrals pro Layer
    return integral;
}, std::plus<std::complex<double>>());