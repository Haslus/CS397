#include "PRNG.h"
#include "NeuralNet.h"
#include <iostream>
/*******************************
	Custom Constructor
*******************************/
CS397::NeuralNet::NeuralNet(const DatasetCreator::Dataset & data, const std::vector<unsigned>& topology, double lr, ActivationFunction::Type function)
{
	mdata = data;
	mtopology = topology;
	mlr = lr;
	mfunction = function;
	for (unsigned i = 0; i < mtopology.size(); i++)
	{
		std::vector<std::vector<double>> a;

		for (unsigned j = 0; j < mtopology[i] + 1; j++)
		{
			std::vector<double > w;
			if (i == mtopology.size() - 1)
			{
				for (unsigned k = 0; k < mtopology.back(); k++)
				{
					a.push_back(std::vector<double>());
				}
				break;
			}
			else
			{

				for(unsigned k = 0; k < mtopology[i + 1]; k++)
					w.push_back(PRNG::RandomNormalizedDouble());
			}

			a.push_back(w);
		}

		mweigths.push_back(a);
	}
}
/*******************************
	Forward propagation, returns an output
*******************************/
std::vector<double> CS397::NeuralNet::ForwardPropagation(const std::vector<double>& input)
{

	std::vector<double> prev = input;
	prev.push_back(1);

	double result = 0;

	//For each final Y
	for (unsigned i = 0; i < mweigths.size(); i++)
	{
		if (i == mweigths.size() - 1)
			break;
		else
		{
			prev[prev.size() - 1] = 1;
		}
		//Final layer
		std::vector<double> next(mweigths[i + 1].size(),0);
		for (unsigned j = 0; j < mweigths[i].size(); j++)
		{
			for (unsigned k = 0; k < mweigths[i][j].size(); k++)
			{
				next[k] += prev[j] * mweigths[i][j][k];
			}

		}
		//Apply function
		for (auto & value : next)
		{
			switch (mfunction)
			{
			case ActivationFunction::Type::eSigmoid:
			{
				value = Sigmoid(value);
				break;
			}
			case ActivationFunction::Type::eTanh:
			{
				value = Tanh(value);
				break;
			}
			}
		}

		prev = next;

		
	}

	return prev;

}
/*******************************
	Iterate once
*******************************/
void CS397::NeuralNet::Iteration()
{

	for (int i = 0; i < mdata.first.size(); i++)
	{
		std::vector<double> output = ForwardPropagation(mdata.first[i]);
		BackPropagation(mdata.first[i],output, mdata.second[i]);
	}


}
/*******************************
	Get weights
*******************************/
CS397::NetworkWeights CS397::NeuralNet::GetWeights() const
{
	return mweigths;
}
/*******************************
	Get cost of Neural Network
*******************************/
double CS397::NeuralNet::Cost(const DatasetCreator::Dataset & data)
{
	double cost = 0;
	for (int i = 0; i < data.first.size(); i++)
	{
		auto output = ForwardPropagation(data.first[i]);

		for (int j = 0; j < output.size(); j++)
		{
			cost += pow(data.second[i][j] - output[j],2);
			
		}

	}
	return cost  / (2.0 * data.first.size());
}
/*******************************
	Sigmoid function
*******************************/
double CS397::NeuralNet::Sigmoid(const double & x)
{
	return 1.0 / (1.0 + exp(-x));
}

/*******************************
	Tanh fuction
*******************************/
double CS397::NeuralNet::Tanh(const double & x)
{
	return 2.0 * Sigmoid(2.0 * x) - 1.0;
}

/*******************************
	Backpropagate the error and fix it
*******************************/
void CS397::NeuralNet::BackPropagation(const std::vector<double> & input,const std::vector<double>& output, const std::vector<double> & real_output)
{
	std::vector<std::vector<double>> values = ValuesBeforeFunction(input);
	std::vector<std::vector<std::vector<double>>> gradients(mweigths.size());

	NetworkWeights weights = mweigths;
	//For each layer
	for (unsigned i = static_cast<int>(mweigths.size() - 2); i >= 0; i--)
	{
		std::vector<std::vector<double>> layer_grad;
		if (i == mweigths.size() - 2)
		{
			for (unsigned j = 0; j < mweigths[i].size(); j++)
			{
				std::vector<double> gradient;
				for (unsigned k = 0; k < mweigths[i][j].size(); k++)
				{

					double alpha = -(real_output[k] - output[k]);

					double beta = 0;

					switch (mfunction)
					{
					case ActivationFunction::Type::eSigmoid:
					{
						beta = output[k] * ( 1.0 - output[k]);
						break;
					}
					case ActivationFunction::Type::eTanh:
					{
						beta = 1.0 - pow(output[k], 2);
						break;
					}
					}

					double gamma = values[i][j];

					gradient.push_back(alpha * beta);

					weights[i][j][k] = weights[i][j][k] - mlr *alpha * beta * gamma;
				}

				layer_grad.push_back(gradient);
			}
		}
		else
		{
			for (unsigned j = 0; j < mweigths[i].size(); j++)
			{
				std::vector<double> gradient;
				for (unsigned k = 0; k < mweigths[i][j].size(); k++)
				{
					double alpha = 0;

					for (int l = 0; l < gradients[i + 1][k].size(); l++)
					{
						alpha += gradients[i + 1][k][l] * mweigths[i + 1][k][l];
					}

					double beta = 0;

					switch (mfunction)
					{
					case ActivationFunction::Type::eSigmoid:
					{
						beta = values[i + 1][k] * (1.0 - values[i + 1][k]);
						break;
					}
					case ActivationFunction::Type::eTanh:
					{
						beta = 1.0 - pow(values[i + 1][k], 2);
						break;
					}
					}

					double gamma = values[i][j];

					gradient.push_back(alpha * beta);

					weights[i][j][k] = weights[i][j][k] - mlr * alpha * beta * gamma;
				}

				layer_grad.push_back(gradient);
			}
				

			

		}

		gradients[i] = layer_grad;

		if( i == 0)
			break;
	}

	mweigths = weights;
}
/*******************************
	Similar to ForwardPropagation
	but stores the value of each node
*******************************/
std::vector<std::vector<double>> CS397::NeuralNet::ValuesBeforeFunction(const std::vector<double>& input)
{

	std::vector<double> prev = input;
	prev.push_back(1);

	std::vector<std::vector<double>> values;

	double result = 0;

	//For each final Y
	for (unsigned i = 0; i < mweigths.size(); i++)
	{
		if (i == mweigths.size() - 1)
			break;
		else
		{
			prev[prev.size() - 1] = 1;
		}
		//Final layer
		std::vector<double> next(mweigths[i + 1].size(), 0);
		values.push_back(std::vector<double>(mweigths[i].size()));
		for (unsigned j = 0; j < mweigths[i].size(); j++)
		{
			
			values[i][j] = prev[j];
			for (unsigned k = 0; k < mweigths[i][j].size(); k++)
			{
				next[k] += prev[j] * mweigths[i][j][k];
			}

			
		}
	

		for (auto & value : next)
		{
			switch (mfunction)
			{
			case ActivationFunction::Type::eSigmoid:
			{
				value = Sigmoid(value);
				break;
			}
			case ActivationFunction::Type::eTanh:
			{
				value = Tanh(value);
				break;
			}
			}
		}

		prev = next;


	}

	return values;
}
