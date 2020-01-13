#include "Regression.h"

CS397::Regression::Regression(const Dataset & dataset, const std::vector<Feature>& features, double lr, bool meanNormalization)
{
	//Normalize if needed
	if (meanNormalization)
	{

	}

	this->features = features;
	this->dataset = dataset;
	this->lr = lr;
}

double CS397::Regression::Predict(const std::vector<double>& input) const
{
	double result = 0;
	for (unsigned i = 0; i < features.size(); i++)
	{
		int index = features[i].inputIdx;
		int pow = features[i].power;
		double theta = features[i].theta;

		if (index == -1)
			result += theta;
		else
			result += std::pow(input[index], pow) * theta;

		
	}
	return result;
}

std::vector<double> CS397::Regression::Predict(const std::vector<  std::vector<double>>& input) const
{
	std::vector<double> result;
	for (auto inp : input)
	{
		result.push_back(Predict(inp));
	}

	return   result;
}

double CS397::Regression::Cost(const std::vector<double>& output, const std::vector<double>& target) const
{
	double value = 0;

	for (unsigned i = 0; i < output.size(); i++)
	{
		value += std::pow(output[i] - target[i], 2);
	}

	value = static_cast<float>(value / output.size());

	return value;

}

void CS397::Regression::Iteration()
{
}
