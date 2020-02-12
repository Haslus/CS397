#include "Clustering.h"

/***********************************************

	Custom Constructor

***********************************************/
CS397::KMeans::KMeans(const Dataset & data, const std::vector<std::vector<double>>& initialCentroids, bool meanNormalization)
{
	this->m_data = data;
	this->m_initialCentroids = this->m_currentCentroids = initialCentroids;
	this->m_meanNormalization = meanNormalization;

	this->m_number_of_clusters = this->m_initialCentroids.size();
	if (m_meanNormalization)
	{

	}

	
}
/***********************************************

	Predict a dataset

***********************************************/
std::vector<unsigned> CS397::KMeans::Predict(const Dataset & input) const
{
	std::vector<unsigned> result;
	//For each sample of the Dataset
	for (int i = 0; i < input.size(); i++)
	{
		result.push_back(Predict(input[i]));
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
	for (int c = 0; c < m_number_of_clusters; c++)
	{
		double current_result = 0;
		for (int i = 0; i < input.size(); i++)
		{
			current_result += input[i] - this->m_currentCentroids[c][i];
		}
		//If the centroid is near, assign
		if (std::abs(current_result) < result)
		{
			result = std::abs(current_result);
			index = c;
		}

	}

	return index;
}
/***********************************************

	iterate to learn

***********************************************/
void CS397::KMeans::Iteration()
{
	//Predict values (Assign closest Centroid)
	m_cluster_index = Predict(m_data);
	//Calculate Cost
	Cost();
	//Assign new centroids
	CalculateClusters();

}

double CS397::KMeans::Cost()
{
	float total_cost = 0;
	//Iterate dataset
	for (int i = 0; i < m_data.size(); i++)
	{
		const std::vector<double> & sample = m_data[i];
		float cost = 0;
		//Distortion function
		for (int j = 0; j < sample.size(); j++)
		{
			cost += sample[j] - m_currentCentroids[m_cluster_index[i]][j];
		}

		cost *= cost;
		total_cost += cost;

	}

	total_cost /= m_data.size();

	return total_cost;
}

double CS397::KMeans::Cost(const Dataset & input)
{
	m_cluster_index = Predict(input);

	float total_cost = 0;
	for (int i = 0; i < input.size(); i++)
	{
		const std::vector<double> & sample = input[i];
		float cost = 0;
		//Distortion function
		for (int j = 0; j < sample.size(); j++)
		{
			cost += sample[j] - m_currentCentroids[m_cluster_index[i]][j];
		}

		cost *= cost;
		total_cost += cost;

	}

	total_cost /= input.size();

	return total_cost;
}

void CS397::KMeans::CalculateClusters()
{
	//Cluster->Samples->Sample Data
	std::vector<std::vector<std::vector<double>>> result;
	result.resize(m_initialCentroids.size());
	for (int i = 0; i < m_cluster_index.size(); i++)
	{
		int index = m_cluster_index[i];

		std::vector<double> sample = m_data[i];

		result[index].push_back(sample);
	}

	for (int i = 0; i < result.size(); i++)
	{
		std::vector<double> average;
		average.resize(m_data[0].size());
		for (int j = 0; j < result[i].size(); j++)
		{
			for (int k = 0; k < result[i][j].size(); k++)
			{
				average[j] += result[i][j][k];
			}

		}

		for (int k = 0; k < m_data[0].size(); k++)
		{
			average[k] /= result[i].size();
		}

		m_currentCentroids[i] = average;

	}
}

CS397::FuzzyCMeans::FuzzyCMeans(const Dataset & data, const std::vector<std::vector<double>>& initialCentroids, double fuzziness, bool meanNormalization)
{
}

std::vector<std::vector<double>> CS397::FuzzyCMeans::Predict(const Dataset & input) const
{
	return std::vector<std::vector<double>>();
}

std::vector<double> CS397::FuzzyCMeans::Predict(const std::vector<double>& input) const
{
	return std::vector<double>();
}

void CS397::FuzzyCMeans::Iteration()
{
}

double CS397::FuzzyCMeans::Cost()
{
	return 0.0;
}

double CS397::FuzzyCMeans::Cost(const Dataset & input)
{
	return 0.0;
}
