//---------------------------------------------------------------------------
#ifndef NEURALNET_H
#define NEURALNET_H
//---------------------------------------------------------------------------


#include "ActivationFunctions.h"
#include "DatasetCreator.h"

#include <vector>
#include <functional>


namespace CS397
{

using NetworkWeights = std::vector<std::vector<std::vector<double>>>;

class NeuralNet
{
public:
    // constructor where the data to train the neural network is passed
    // it will initialize every neuron with a random weight on the range [-1,1]
	NeuralNet(  const DatasetCreator::Dataset & data,                                   // dataset to train the network
                const std::vector<unsigned> & topology,                                 // number of layers and amount on neurons in each layer
                double lr,                                                              // learning rate
                ActivationFunction::Type function = ActivationFunction::Type::eSigmoid  // activation function
                );

    // given an input value, it will propagate those values through the network and produce an output
    // the prediction function of the neural network
	std::vector<double> ForwardPropagation(const std::vector<double> & input);
    
    // produces a train interation with the whole dataset, all weights will be updated
    // first it will produce an output with the forward propagation and the it will back propagate the error
    // updating the weights
	void Iteration();
	
    // returns every weight in the neural network (mainly to check random initialization
	NetworkWeights GetWeights() const;
    
    // computes the cost of the network for the provided dataset
	double Cost(const DatasetCreator::Dataset & data);

	//Student Stuff
	double Sigmoid(const double & x);
	double SigmoidDerivative(const double & x);
	double Tanh(const double & x);
	double TanhDerivative(const double & x);
	double ReLU(const double& x);

	void BackPropagation(const std::vector<double> & input,const std::vector<double> & output, const std::vector<double> & real_output);

	DatasetCreator::Dataset mdata;           // dataset to train the network
	std::vector<unsigned> mtopology;           // number of layers and amount on neurons in each layer
	double mlr;                                  // learning rate
	ActivationFunction::Type mfunction;		 // activation function

	NetworkWeights mweigths;

	double SubstractVectors(const std::vector<double> & a, const std::vector<double> & b);
	std::vector<std::vector<double>> ValuesBeforeFunction(const std::vector<double> & input);
};

}
#endif