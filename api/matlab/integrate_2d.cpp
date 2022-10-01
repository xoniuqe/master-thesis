/*
* arrayProduct.cpp - example in MATLAB External Interfaces
*
* Multiplies an input scalar (multiplier)
* times a MxN matrix (inMatrix)
* and outputs a MxN matrix (outMatrix)
*
* Usage : from MATLAB
*         >> outMatrix = arrayProduct(multiplier, inMatrix)
*
* This is a C++ MEX-file for MATLAB.
* Copyright 2017 The MathWorks, Inc.
* GetInt2D(A,b,r,mu,k,lagPoints,resY_std);
*/


#include "mex.hpp"
#include "mexAdapter.hpp"

#include <armadillo>
#include <steepest_descent/configuration.h>
#include <steepest_descent/integral_2d.h>

class MexFunction : public matlab::mex::Function {
    std::ostringstream stream;

public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        //checkArguments(outputs, inputs);

        auto  A = to_armadillo_matrix(std::move(inputs[0]));

        auto b = to_vec3(std::move(inputs[1]));


        auto r = to_vec3(std::move(inputs[2]));

        auto theta = to_vec3(std::move(inputs[3]));


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

private:
    inline arma::mat to_armadillo_matrix(const matlab::data::TypedArray<double>& matrix) 
    {
        return std::move(arma::mat{ {matrix[0][0], matrix[0][1]}, {matrix[1][0], matrix[1][1] }, {matrix[2][0], matrix[2][1] }});
    }
    inline arma::vec3 to_vec3(const matlab::data::TypedArray<double>& vector) {
        return std::move(arma::vec3{ vector[0], vector[1], vector[2] });
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

    void arrayProduct(matlab::data::TypedArray<double>& inMatrix, double multiplier) {

        for (auto& elem : inMatrix) {
            elem *= multiplier;
        }
    }

    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        matlab::data::ArrayFactory factory;

        if (inputs.size() != 2) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Two inputs required") }));
        }

        auto dimensions = inputs[0].getDimensions();

       // dimensions.
       // if (inputs[0].getDimensions().size() != 3 || inputs[0].getDimensions()[0].Get

        if (inputs[0].getNumberOfElements() != 1) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input multiplier must be a scalar") }));
        }

        if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[0].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input multiplier must be a noncomplex scalar double") }));
        }

        if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[1].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input matrix must be type double") }));
        }

        if (inputs[1].getDimensions().size() != 2) {
            matlabPtr->feval(u"error",
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input must be m-by-n dimension") }));
        }
    }
};