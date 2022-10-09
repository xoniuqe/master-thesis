auto& [sp1, sp2] = split_points;

//Berechnung von $I(k,y,a,a_1)$ und $I(k,y,b_1,b)$
auto I1 = steepest_desc(left_split, sp1);
auto I2 =  steepest_desc(sp2, right_split);


// Berechnung der um die Singularität $I(k,y,a_1,b_1)$
auto& local_k = this->config.wavenumber_k;
auto green_fun = [k=local_k, y=y, &A, &b, &r, q, s](const double x) -> auto { 
    auto Px = math_utils::calculate_P_x(x, y, A, b, r);
    auto sqrtPx = std::sqrt(Px);
    auto res =  std::exp(1.i * k * (sqrtPx + q * x + s)) * (1. / sqrtPx);
    return res;
};
auto x = integrator->operator()(green_fun, sp1, sp2);
// $I(k,y,a,a_1) + I(k,y,a_1,b_1) +I(k,y,b_1,b)$
return I1 + x + I2;