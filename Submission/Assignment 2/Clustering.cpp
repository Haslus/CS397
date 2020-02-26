#include "Clustering.h"
#include <iostream>
#include <fstream>
#include <string>
/***********************************************

	Custom Constructor

***********************************************/
CS397::KMeans::KMeans(const Dataset & data, const std::vector<std::vector<double>>& initialCentroids, bool meanNormalization)
{
	this->m_data = data;
	this->m_initialCentroids = initialCentroids;
	this->m_currentCentroids = initialCentroids;
	this->m_meanNormalization = meanNormalization;

	this->m_number_of_clusters = static_cast<unsigned int>(this->m_initialCentroids.size());

	if (m_meanNormalization)
	{
		for (int f = 0; f < m_data[0].size(); f++)
		{
			double min = DBL_MAX, max = -DBL_MAX, mean = 0;

			for (int i = 0; i < m_data.size(); i++)
			{
				double value = m_data[i][f];

				mean += value;

				if (value < min)
					min = value;

				if (value > max)
					max = value;
			}

			mean /= m_data.size();

			double delta = max - min;
			m_norm_values.push_back({delta,mean});

			for (int i = 0; i < m_data.size(); i++)
			{
				double value = m_data[i][f];

				m_data[i][f] = (value - mean) / delta;
			}

			for (int c = 0; c < m_currentCentroids.size(); c++)
			{
				m_currentCentroids[c][f] = (m_currentCentroids[c][f] - mean) / delta;
			}
		}
	}

	
}
/***********************************************

	Predict a dataset

***********************************************/
std::vector<unsigned> CS397::KMeans::Predict(const Dataset & input) const
{
	std::vector<unsigned> result;
	Dataset input_to_predict = input;

	if (m_meanNormalization)
	{
		for (int f = 0; f < m_data[0].size(); f++)
		{
			auto values = m_norm_values[f];

			for (int i = 0; i < m_data.size(); i++)
			{
				double value = input[i][f];

				input_to_predict[i][f] = (value - values.mean) / values.delta;
			}
		}
	}
	//For each sample of the Dataset
	for (int i = 0; i < input_to_predict.size(); i++)
	{
		result.push_back(Predict(input_to_predict[i]));
	}

	return result;
}
/***********************************************

	Predict a sample

***********************************************/
unsigned CS397::KMeans::Predict(const std::vector<double>& input) const
{
	//For each value of the sample
	unsigned index = -1;
	double result = DBL_MAX;
	//Iterate by number of clusters
	for (unsigned c = 0; c < m_number_of_clusters; c++)
	{
		double current_result = 0;
		for (unsigned i = 0; i < input.size(); i++)
		{
			double value = (input[i] - this->m_currentCentroids[c][i]);
			current_result += value * value;
		}

		//If the centroid is near, assign
		if (current_result <= result)
		{
			result = current_result;
			index = c;
		}

	}

	return index;
}
/***********************************************

	Iterate to learn

***********************************************/
void CS397::KMeans::Iteration()
{
	//Predict values (Assign closest Centroid)
	if (m_meanNormalization)
		m_cluster_index = Normalized_Predict();
	else
		m_cluster_index = Predict(m_data);
	//Assign new centroids
	CalculateClusters();

}
/***********************************************

	Get cost of the training set

***********************************************/
double CS397::KMeans::Cost()
{
	double total_cost = 0;
	//Iterate dataset
	for (int i = 0; i < m_data.size(); i++)
	{
		const std::vector<double> & sample = m_data[i];
		double cost = 0;
		//Distortion function
		for (int j = 0; j < sample.size(); j++)
		{
			double value = sample[j] - m_currentCentroids[m_cluster_index[i]][j];
			cost += value * value;
		}

		total_cost += cost;

	}

	total_cost /= m_data.size();

	return total_cost;
}
/***********************************************

	Get cost of the test set

***********************************************/
double CS397::KMeans::Cost(const Dataset & input)
{
	m_cluster_index = Predict(input);

	double total_cost = 0;
	for (int i = 0; i < input.size(); i++)
	{
		const std::vector<double> & sample = input[i];
		double cost = 0;
		//Distortion function
		for (int j = 0; j < sample.size(); j++)
		{
			double value = sample[j] - m_currentCentroids[m_cluster_index[i]][j];
			cost += value * value;
		}

		total_cost += cost;

	}

	total_cost /= input.size();

	return total_cost;
}
/***********************************************

	Recalculate new positions for the centroids

***********************************************/
void CS397::KMeans::CalculateClusters()
{
	//Cluster->Samples->Sample Data
	std::vector<std::vector<std::vector<double>>> result;
	result.resize(m_initialCentroids.size());

	for (int i = 0; i < m_cluster_index.size(); i++)
	{
		int index = m_cluster_index[i];

		std::vector<double> & sample = m_data[i];

		result[index].push_back(sample);
	}

	for (int i = 0; i < result.size(); i++)
	{
		std::vector<double> average;
		for (int j = 0; j < m_data[0].size(); j++)
		{
			double current_average = 0;

			for (int k = 0; k < result[i].size(); k++)
			{		
				current_average += result[i][k][j];

			}

			current_average /= result[i].size();

			average.push_back(current_average);
		}

		m_currentCentroids[i] = average;
	}
}
/***********************************************

	Write the clusters into a .csv

***********************************************/
void CS397::KMeans::OutputClusters(Dataset dataset)
{
	//Write clusters as strings
	

	std::vector<unsigned int> prediction = Predict(dataset);

	std::vector<std::string> text_clusters;

	for (int i = 0; i < prediction.size(); i++)
	{

			text_clusters.push_back(std::to_string(prediction[i]));
		
	}

	std::ofstream outfile;
	outfile.open("OutIris.csv");
	if (outfile.is_open())
	{
		outfile << "ID,cluster\n";
		for (int i = 0; i < text_clusters.size(); i++)
		{
			outfile << std::to_string(i + 1);
			outfile << ',';

			outfile << text_clusters[i];
			outfile << ',';
			
			outfile << '\n';
		}
		


		outfile.close();
	}

}

/***********************************************

	Predict for already normalized data

***********************************************/
std::vector<unsigned> CS397::KMeans::Normalized_Predict() const
{
	std::vector<unsigned> result;

	//For each sample of the Dataset
	for (int i = 0; i < m_data.size(); i++)
	{
		result.push_back(Predict(m_data[i]));
	}

	return result;
}
/***********************************************

	Custom constructor

***********************************************/
CS397::FuzzyCMeans::FuzzyCMeans(const Dataset & data, const std::vector<std::vector<double>>& initialCentroids, double fuzziness, bool meanNormalization)
{
	m_data = data;
	m_initialCentroids =  initialCentroids;
	m_currentCentroids = initialCentroids;
	m_fuzziness = fuzziness;
	m_meanNormalization = meanNormalization;
	m_row_clusters = static_cast<unsigned int>(m_initialCentroids.size());
	m_column_samples = static_cast<unsigned int>(m_data.size());
	ProbabilityMatrix = InitialProbabilityMatrix(m_row_clusters, m_column_samples);

	if (m_meanNormalization)
	{
		for (int f = 0; f < m_data[0].size(); f++)
		{
			double min = -DBL_MAX, max = DBL_MAX, mean = 0;

			for (int i = 0; i < m_data.size(); i++)
			{
				double value = m_data[i][f];

				mean += value;

				if (value < min)
					min = value;

				if (value > max)
					max = value;
			}

			mean /= m_data.size();

			double delta = max - min;
			m_norm_values.push_back({ delta,mean });

			for (int i = 0; i < m_data.size(); i++)
			{
				double value = m_data[i][f];

				m_data[i][f] = (value - mean) / delta;
			}


			for (int c = 0; c < m_currentCentroids.size(); c++)
			{
				m_currentCentroids[c][f] = (m_currentCentroids[c][f] - mean) / delta;
			}
		}
	}


}
/***********************************************

	Predict for all samples

***********************************************/
std::vector<std::vector<double>> CS397::FuzzyCMeans::Predict(const Dataset & input) const
{
	std::vector<std::vector<double>> input_to_predict = input;
	std::vector<std::vector<double>> result;
	if (m_meanNormalization)
	{
		for (int f = 0; f < m_data[0].size(); f++)
		{
			auto values = m_norm_values[f];

			for (int i = 0; i < m_data.size(); i++)
			{
				double value = input[i][f];

				input_to_predict[i][f] = (value - values.mean) / values.delta;
			}
		}
	}

	for (int i = 0; i < input.size(); i++)
	{
		result.push_back(Predict(input_to_predict[i]));
	}

	return result;
}
/***********************************************

	Predict a sample

***********************************************/
std::vector<double> CS397::FuzzyCMeans::Predict(const std::vector<double>& input) const
{
	std::vector<double> weight_per_cluster;

	for (unsigned k = 0; k < m_row_clusters; k++)
	{
		double top_value = 0;
		for (unsigned i = 0; i < input.size(); i++)
		{
			double value = input[i] - m_currentCentroids[k][i];
			top_value += value * value;
		}

		double final_result = 0;
		top_value = std::sqrt(top_value);
		for (unsigned j = 0; j < m_row_clusters; j++)
		{

			double bot_value = 0;

			for (int i = 0; i < input.size(); i++)
			{
				double value = input[i] - m_currentCentroids[j][i];
				bot_value += value * value;
			}

			if (bot_value == 0)
			{
				
				weight_per_cluster = std::vector<double>(m_row_clusters, 0);
				weight_per_cluster[j] = 1.0;
				return weight_per_cluster;
				
			}

			final_result += std::pow(top_value / std::sqrt(bot_value), 2.0 / (m_fuzziness - 1.0));
		}

		weight_per_cluster.push_back(1.0 / final_result);
	}

	return weight_per_cluster;
}
/***********************************************

	Iterate to learn

***********************************************/
void CS397::FuzzyCMeans::Iteration()
{
	UpdateCentroids();
	std::vector<std::vector<double>> values;
	if(m_meanNormalization)
		values = Normalized_Predict();
	else
		values = Predict(m_data);

	ProbabilityMatrix = StoreProbabilityMatrix(values);
}
/***********************************************

	Cost from the training set

***********************************************/
double CS397::FuzzyCMeans::Cost()
{
	double result = 0;
	for (unsigned m = 0; m < m_column_samples; m++)
	{
		for (unsigned k = 0; k < m_row_clusters; k++)
		{

			double w = ProbabilityMatrix[k * m_column_samples + m];

			std::vector<double> sample = m_data[m];

			double sample_cost = 0;
			for (unsigned s = 0; s < sample.size(); s++)
			{
				double value = sample[s] - m_currentCentroids[k][s];
				sample_cost += value * value;
			}

			result += pow(w, m_fuzziness) * (sample_cost);

		}
	}

	return static_cast<float>(result / m_data.size());
}
/***********************************************

	Cost from a given input

***********************************************/
double CS397::FuzzyCMeans::Cost(const Dataset & input)
{
	std::vector<std::vector<double>> values = Predict(input);
	std::vector<double> matrix = StoreProbabilityMatrix(values);
	
	double result = 0;
	for (unsigned m = 0; m < input.size(); m++)
	{
		for (unsigned k = 0; k < m_row_clusters; k++)
		{
		
			double w = matrix[k * input.size() + m];

			std::vector<double> sample = input[m];

			double sample_cost = 0;
			for (int s = 0; s < sample.size(); s++)
			{
				double value = sample[s] - m_currentCentroids[k][s];
				sample_cost += value * value;
			}

			result += pow(w, m_fuzziness) * (sample_cost);

		}
	}

	return static_cast<float>(result / input.size());
}
/***********************************************

	Update new positions for the centroids

***********************************************/
void CS397::FuzzyCMeans::UpdateCentroids()
{
	for (unsigned k = 0; k < m_row_clusters; k++)
	{
		
		for (unsigned s = 0; s < m_data[0].size(); s++)
		{
			double top_value = 0;
			double bot_value = 0;

			for (unsigned i = 0; i < m_column_samples; i++)
			{
				double w = ProbabilityMatrix[m_column_samples * k + i];
				w = std::pow(w, m_fuzziness);

				top_value += w * m_data[i][s];
				bot_value += w;
			}

			m_currentCentroids[k][s] = top_value / bot_value;

		}

	}

}
/***********************************************

	Convert std::vector/vector/double to
	std::vector/double

***********************************************/
std::vector<double> CS397::FuzzyCMeans::StoreProbabilityMatrix(const std::vector<std::vector<double>> & values)
{
	std::vector<double> matrix;
	for (unsigned k = 0; k < m_row_clusters; k++)
	{
		for (unsigned i = 0; i < values.size(); i++)
		{
			matrix.push_back(values[i][k]);
		}
	}

	return matrix;
}
/***********************************************

	Predict for normalized input

***********************************************/
std::vector<std::vector<double>> CS397::FuzzyCMeans::Normalized_Predict() const
{

	std::vector<std::vector<double>> result;

	for (int i = 0; i < m_data.size(); i++)
	{
		result.push_back(Predict(m_data[i]));
	}

	return result;
}
