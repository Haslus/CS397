#include "PRNG.h"
#include "NeuralNet.h"

CS397::NeuralNet::NeuralNet(const DatasetCreator::Dataset & data, const std::vector<unsigned>& topology, double lr, ActivationFunction::Type function)
{
	mdata = data;
	mtopology = topology;
	mlr = lr;
	mfunction = function;
	for (int i = 0; i < mtopology.size(); i++)
	{
		std::vector<std::vector<double>> a;

		for (int j = 0; j < mtopology[i] + 1; j++)
		{
			std::vector<double > w;
			if (i == mtopology.size() - 1)
			{
				for (int k = 0; k < mtopology.back(); k++)
				{
					a.push_back(std::vector<double>());
				}
				break;
			}
			else
			{

				for(int k = 0; k < mtopology[i + 1]; k++)
					w.push_back(PRNG::RandomNormalizedDouble());
			}

			a.push_back(w);
		}

		mweigths.push_back(a);
	}
}

std::vector<double> CS397::NeuralNet::ForwardPropagation(const std::vector<double>& input)
{

	//For each final Y
	std::vector<double> prev = input;
	prev.push_back(1);

	//For each final outputs
//	for (int i = 0; i < mtopology.back(); i++)
	//{
	double result = 0;

	for (unsigned i = 0; i < mweigths.size(); i++)
	{
		if (i == mweigths.size() - 1)
			return prev;
		else
		{
			prev[prev.size() - 1] = 1;
		}
		//Final layer
		std::vector<double> next(mweigths[i + 1].size(),0);
		for (unsigned j = 0; j < mweigths[i].size(); j++)
		{
			//double next_value = 0;
			for (unsigned k = 0; k < mweigths[i][j].size(); k++)
			{
				next[k] += prev[j] * mweigths[i][j][k];
			}

			//next.push_back(next_value);
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
}

void CS397::NeuralNet::Iteration()
{

	for (int i = 0; i < mdata.first.size(); i++)
	{
		std::vector<double> output = ForwardPropagation(mdata.first[i]);
		BackPropagation(mdata.first[i],output, mdata.second[i]);
	}


}

CS397::NetworkWeights CS397::NeuralNet::GetWeights() const
{
	return mweigths;
}

double CS397::NeuralNet::Cost(const DatasetCreator::Dataset & data)
{
	double cost = 0;
	for (int i = 0; i < data.first.size(); i++)
	{
		auto output = ForwardPropagation(data.first[i]);
		double value = 0;
		for (int j = 0; j < output.size(); j++)
		{
			value += output[j] - data.second[i][j];
		}

		cost += pow(value, 2);
	}
	return cost  / (2.0 * data.first.size());
}

double CS397::NeuralNet::Sigmoid(const double & x)
{
	return 1.0 / (1.0 + exp(-x));
}

double CS397::NeuralNet::SigmoidDerivative(const double & x)
{
	return Sigmoid(x) * (1.0 - Sigmoid(x));
}

double CS397::NeuralNet::Tanh(const double & x)
{
	return 2.0 * Sigmoid(2.0 * x) - 1.0;
}

double CS397::NeuralNet::TanhDerivative(const double & x)
{
	return 1 - pow(tanh(x), 2);
}

double CS397::NeuralNet::ReLU(const double & x)
{
	return std::fmax(0, x);
}

void CS397::NeuralNet::BackPropagation(const std::vector<double> & input,const std::vector<double>& output, const std::vector<double> & real_output)
{
	std::vector<std::vector<double>> values = ValuesBeforeFunction(input);

	for (unsigned i = mweigths.size() - 2; i >= 0; i--)
	{
		if (i == mweigths.size() - 2)
		{
			for (unsigned j = 0; j < mweigths[i].size(); j++)
			{

				for (unsigned k = 0; k < mweigths[i][j].size(); k++)
				{

					double alpha = -(real_output[k] - output[k]);

					double beta = 1.0 - pow(output[k], 2);

					double gamma = values[i][j];

					mweigths[i][j][k] -= mlr * (alpha * beta * gamma);
				}
			}
		}
		else
		{
			for (unsigned j = 0; j < mweigths[i].size(); j++)
			{
				for (unsigned k = 0; k < mweigths[i][j].size(); k++)
				{
					double alpha = 0;

					for (int l = 0; l < output.size(); l++)
					{
						alpha += -(real_output[l] - output[l]) * (1.0 - pow(output[l], 2)) * mweigths[i + 1][k][l];
					}

					double beta = 1.0 - pow(values[i + 1][k], 2);

					double gamma = values[i][j];

					mweigths[i][j][k] -= mlr * (alpha * beta * gamma);
				}
			}
				

			

		}

		if (i == 0)
			return;

	}
}

double CS397::NeuralNet::SubstractVectors(const std::vector<double>& a, const std::vector<double>& b)
{
	double value = 0;
	for (int i = 0; i < a.size(); i++)
	{
		value += a[i] - b[i];
	}
	return value;
}

std::vector<std::vector<double>> CS397::NeuralNet::ValuesBeforeFunction(const std::vector<double>& input)
{

	//For each final Y
	std::vector<double> prev = input;
	prev.push_back(1);

	std::vector<std::vector<double>> values;
	//For each final outputs
//	for (int i = 0; i < mtopology.back(); i++)
	//{
	double result = 0;

	for (unsigned i = 0; i < mweigths.size(); i++)
	{
		if (i == mweigths.size() - 1)
			return values;
		else
		{
			prev[prev.size() - 1] = 1;
		}
		//Final layer
		std::vector<double> next(mweigths[i + 1].size(), 0);
		values.push_back(std::vector<double>(mweigths[i].size()));
		for (unsigned j = 0; j < mweigths[i].size(); j++)
		{
			//double next_value = 0;
			values[i][j] = prev[j];
			for (unsigned k = 0; k < mweigths[i][j].size(); k++)
			{
				next[k] += prev[j] * mweigths[i][j][k];
			}

			//next.push_back(next_value);
		}
	//	values.push_back(next);

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
}
