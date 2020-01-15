#include "Regression.h"
#include <iostream>

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

	return result;
}

double CS397::Regression::Cost(const std::vector<double>& output, const std::vector<double>& target) const
{
	double value = 0;

	for (unsigned i = 0; i < output.size(); i++)
	{
		double current_value = output[i] - target[i];
		value += current_value * current_value;
	}

	value = static_cast<double>(value / (2.f * output.size()));

	return value;

}

void CS397::Regression::Iteration()
{

	std::vector<double> prediction = Predict(dataset.first);
	//std::cout << Cost(prediction, dataset.second) << std::endl;
	Cost_Derivative(prediction, dataset.second);
}

double CS397::Regression::Cost_Derivative(const std::vector<double>& output, const std::vector<double>& target)
{

	std::vector<double> new_thetas;


	for (unsigned j = 0; j < features.size(); j++)
	{
		int index = features[j].inputIdx;
		int pow = features[j].power;

		double value = 0;
		for (unsigned i = 0; i < output.size(); i++)
		{
			double current_value = 0;
			current_value = (output[i] - target[i]);
			if (index != -1)
				current_value *= std::pow(dataset.first[i][index],pow);

			value += current_value;

		}

		value = static_cast<double>(value / output.size());
		new_thetas.push_back(value);
	
	}

	for (unsigned j = 0; j < features.size(); j++)
	{
		features[j].theta = features[j].theta - lr  * new_thetas[j];
	}
	


	return 0.0;
}
