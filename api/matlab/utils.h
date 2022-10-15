#pragma once


#include "mex.hpp"
#include <armadillo>

namespace utils {

    inline arma::mat to_armadillo_matrix(const matlab::data::TypedArray<double>& matrix)
    {
        return std::move(arma::mat{ {matrix[0][0], matrix[0][1]}, {matrix[1][0], matrix[1][1] }, {matrix[2][0], matrix[2][1] } });
    }
    inline arma::vec3 to_vec3(const matlab::data::TypedArray<double>& vector) {
        return std::move(arma::vec3{ vector[0], vector[1], vector[2] });
    }

 }