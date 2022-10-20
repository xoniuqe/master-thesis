#include "mex.hpp"
#include "mexAdapter.hpp"
#include "utils.h"
#include <armadillo>
#include <steepest_descent/configuration.h>
#include <steepest_descent/integral_1d.h>

class MexFunction : public matlab::mex::Function {
    std::ostringstream stream;
private:
    auto integrate_1d_with_theta(matlab::mex::ArgumentList& inputs) -> std::complex<double> {
        auto  A = utils::to_armadillo_matrix(std::move(inputs[0]));
        auto b = utils::to_vec3(std::move(inputs[1]));
        auto r = utils::to_vec3(std::move(inputs[2]));
        double k = inputs[3][0];
        auto theta = utils::to_vec3(std::move(inputs[4]));
        double y = inputs[5][0];
        double left_split = inputs[6][0];
        double right_split = inputs[7][0];


        matlab::data::TypedArray<double> nodesInput = inputs[8];
        matlab::data::TypedArray<double> weightsInput = inputs[9];

        std::vector<double> nodes(nodesInput.begin(), nodesInput.end());
        std::vector<double> weights(weightsInput.begin(), weightsInput.end());


        config::configuration config;
        config.wavenumber_k = k;

        integrator::gsl_integrator gslintegrator;
        integral::integral_1d integral1d(config, &gslintegrator, nodes, weights);

        return integral1d(A, b, r, theta,y, left_split, right_split);
    }

    auto integrate_1d(matlab::mex::ArgumentList& inputs) ->std::complex<double> {
        auto  A = utils::to_armadillo_matrix(std::move(inputs[0]));
        auto b = utils::to_vec3(std::move(inputs[1]));
        auto r = utils::to_vec3(std::move(inputs[2]));
        double k = inputs[3][0];
        double q = inputs[4][0];
        double s = inputs[5][0];
        double y = inputs[6][0];
        double left_split = inputs[7][0];
        double right_split = inputs[8][0];

        matlab::data::TypedArray<double> nodesInput = inputs[9];
        matlab::data::TypedArray<double> weightsInput = inputs[10];
        std::vector<double> nodes(nodesInput.begin(), nodesInput.end());
        std::vector<double> weights(weightsInput.begin(), weightsInput.end());


        auto croots = math_utils::get_complex_roots(y, A, b, r);
        auto c = std::get<0>(croots);
        auto c_0 = std::get<1>(croots);

        config::configuration config;
        config.wavenumber_k = k;

        integrator::gsl_integrator gslintegrator;
        integral::integral_1d integral1d(config, &gslintegrator, nodes, weights);
        return integral1d(A, b, r, q, s, y, c, c_0, left_split, right_split);
    }

    void displayOnMATLAB(std::ostringstream& stream) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        matlab::data::ArrayFactory factory;

        // Pass stream content to MATLAB fprintf function
        matlabPtr->feval(u"fprintf", 0,
            std::vector<matlab::data::Array>({ factory.createScalar(stream.str()) }));
        // Clear stream buffer
        stream.str("");
    }

public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        //checkArguments(outputs, inputs);

        matlab::data::ArrayFactory factory;

        if (inputs.size() == 10) {
            auto integral = integrate_1d_with_theta(inputs);
            outputs[0] = factory.createScalar(integral);
            return;
        }
        else if (inputs.size() == 11) {
            auto integral = integrate_1d(inputs);
            outputs[0] = factory.createScalar(integral);
            return;
        }
        stream << "Invalid inputs!" << std::endl;
        displayOnMATLAB(stream);
    }
};