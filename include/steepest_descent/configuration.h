#pragma once


namespace config {

	struct configuration {
		double wavenumber_k;
		double tolerance = 0.1;
		size_t gauss_laguerre_nodes;
	};

	struct configuration_2d : public configuration{
		double y_resolution = 0.1;
	};
}