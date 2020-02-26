#include "Regression.h"
#include <iostream>

CS397::Regression::Regression(const Dataset & dataset, const std::vector<Feature>& features, double lr, bool meanNormalization)
{
	Dataset temp = dataset;
	feature_norms.resize(features.size());
	//Normalize if needed
	if (meanNormalization)
	{
		for (unsigned i = 0; i < features.size(); i++)
		{
			int index = features[i].inputIdx;
			if (index == -1)
				continue;
			double stack = 0;
			double max = 0;
			double min = DBL_MAX;
			for (unsigned j = 0; j < dataset.first.size(); j++)
			{
				double value = temp.first[j][index];
				stack += value;

				if (value > max)
					max = value;
				if (value < min)
					min = value;
			}
			stack /= dataset.first.size();
			double delta = max - min;

			feature_norms[i] = (FeatureNorm{ stack,max-min });
			for (unsigned j = 0; j < dataset.first.size(); j++)
			{
				temp.first[j][index] = (temp.first[j][index] - stack) / delta;
			}

		}
	}

	this->features = features;
	this->dataset = temp;
	this->lr = lr;
	this->meanNormalization = meanNormalization;

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
	std::vector<std::vector<double>> norm_input = input;
	//Normalize the passed data if needed
	if (meanNormalization)
	{
		for (unsigned i = 0; i < features.size(); i++)
		{
			int index = features[i].inputIdx;
			if (index == -1)
				continue;

			for (unsigned j = 0; j < norm_input.size(); j++)
			{
				norm_input[j][index] = (norm_input[j][index] - feature_norms[i].mean) / feature_norms[i].range;
			}

		}
	}
	

	for (auto inp : norm_input)
	{
		result.push_back(Predict(inp));
	}

	return result;
}

std::vector<double> CS397::Regression::PredictNormalized(const std::vector<std::vector<double>>& input) const
{
	std::vector<double> result;
	std::vector<std::vector<double>> norm_input = input;

	for (auto inp : norm_input)
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
	std::vector<double> prediction;
	if (!meanNormalization)
	{
		prediction = Predict(dataset.first);
	}
	else
	{
		prediction = PredictNormalized(dataset.first);
	}

	Cost_Derivative(prediction, dataset.second);
}

void CS397::Regression::Cost_Derivative(const std::vector<double>& output, const std::vector<double>& target)
{

	std::vector<double> new_thetas;

	//Calculate the derivative for the new thetas
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
	


	return;
}
