#pragma once

#include <gsl/gsl_vector.h>
namespace gsl_vector_utils
{

	gsl_vector* vector_from_std(std::vector<double> vector, int size) {
		auto _vector = gsl_vector_alloc(size);
		for (auto i = 0; i < size; i++) {
			gsl_vector_set(_vector, i, vector[i]);
		}
		return _vector;
	}

	auto gsl_vector_to_standard(gsl_vector* vector) {
		auto size = vector->size;
		std::vector<double> result(size);
		for (auto i = 0; i < size; i++) {
			result[i] = gsl_vector_get(vector, i);
		}
		return result;
	}
}