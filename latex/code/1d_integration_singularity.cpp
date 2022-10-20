auto& [sp1, sp2] = split_points;

//Berechnung von $I(k,y,a,a_1)$ und $I(k,y,b_1,b)$ (mit Optimierung)
auto I1 = std::abs(left_split - sp1) <= std::numeric_limits<double>::epsilon() ? 
        0. : steepest_desc(left_split) - steepest_desc(sp1);

auto I2 = std::abs(sp2 - right_split) <= std::numeric_limits<double>::epsilon() ? 
        0. : steepest_desc(sp2) - steepest_desc(right_split);

auto& [sp1, sp2] = split_points;

//Berechnung von $I(k,y,a_1,b_1)$ 
auto fun = green_fun_generator(config.wavenumber_k, y, A, b, r, q, s);
auto x = integrator->operator()(fun, sp1, sp2);

// $I(k,y,a,a_1) + I(k,y,a_1,b_1) +I(k,y,b_1,b)$
return I1 + x + I2;