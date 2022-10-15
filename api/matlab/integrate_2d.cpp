#include "mex.hpp"
#include "mexAdapter.hpp"
#include "utils.h"

#include <armadillo>
#include <steepest_descent/configuration.h>
#include <steepest_descent/integral_2d.h>

class MexFunction : public matlab::mex::Function {
    std::ostringstream stream;

public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        //checkArguments(outputs, inputs);

        auto  A = utils::to_armadillo_matrix(std::move(inputs[0]));

        auto b = utils::to_vec3(std::move(inputs[1]));


        auto r = utils::to_vec3(std::move(inputs[2]));

        auto theta = utils::to_vec3(std::move(inputs[3]));


        double k = inputs[4][0];

  
        matlab::data::TypedArray<double> nodesInput = inputs[5];
        matlab::data::TypedArray<double> weightsInput = inputs[6];

        std::vector<double> nodes(nodesInput.begin(), nodesInput.end());
        std::vector<double> weights(weightsInput.begin(), weightsInput.end());


        auto resolution = 0.1;
        if (inputs.size() == 8) {
            resolution = inputs[7][0];
        }

        config::configuration_2d config;
        config.wavenumber_k = k;
        config.y_resolution = resolution;

        integrator::gsl_integrator gslintegrator;
        integrator::gsl_integrator_2d gsl_integrator_2d;
        integral::integral_2d integral2d(config, &gslintegrator, &gsl_integrator_2d, nodes, weights);

        auto integral = integral2d(A, b, r, theta);
        matlab::data::ArrayFactory factory;
        outputs[0] = factory.createScalar(integral);
    }   
};