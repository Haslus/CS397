#include "Clustering.h"

/***********************************************

	Custom Constructor

***********************************************/
CS397::KMeans::KMeans(const Dataset & data, const std::vector<std::vector<double>>& initialCentroids, bool meanNormalization)
{
	this->m_data = data;
	this->m_initialCentroids = initialCentroids;
	this->m_currentCentroids = initialCentroids;
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
			double value = (input[i] - this->m_currentCentroids[c][i]);
			current_result += value * value;
		}

		//If the centroid is near, assign
		if (current_result < result)
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
	m_cluster_index = Predict(m_data);
	//Calculate Cost
	//Cost();
	//Assign new centroids
	CalculateClusters();

}

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

CS397::FuzzyCMeans::FuzzyCMeans(const Dataset & data, const std::vector<std::vector<double>>& initialCentroids, double fuzziness, bool meanNormalization)
{
	m_data = data;
	m_initialCentroids = m_currentCentroids = initialCentroids;

	m_fuzziness = fuzziness;
	m_meanNormalization = meanNormalization;
	m_row_clusters = m_initialCentroids.size();
	m_column_samples = m_data.size();
	ProbabilityMatrix = InitialProbabilityMatrix(m_row_clusters, m_column_samples);


}

std::vector<std::vector<double>> CS397::FuzzyCMeans::Predict(const Dataset & input) const
{
	std::vector<std::vector<double>> total_weights;
	for (int i = 0; i < input.size(); i++)
	{
		total_weights.push_back(Predict(input[i]));
	}

	return total_weights;
}

std::vector<double> CS397::FuzzyCMeans::Predict(const std::vector<double>& input) const
{
	std::vector<double> weight_per_cluster;

	for (int k = 0; k < m_row_clusters; k++)
	{
		double top_value = 0;
		for (int i = 0; i < input.size(); i++)
		{
			double value = input[i] - m_currentCentroids[k][i];
			top_value += value * value;
		}

		double final_result = 0;
		top_value = sqrt(top_value);
		for (int j = 0; j < m_row_clusters; j++)
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
				weight_per_cluster[j] = 1;
				return weight_per_cluster;
				
			}

			final_result += pow((top_value / sqrt(bot_value)), 2.0f / (m_fuzziness - 1.0f));
		}

		weight_per_cluster.push_back(1.0f / final_result);
	}

	//Check for errors
	double total = 0;
	for (auto w : weight_per_cluster)
	{
		total += w;
	}
	if (std::abs(1.0f - total) > 0.1f)
	{
		return std::vector<double>(m_row_clusters, 0);
	}
	else
		return weight_per_cluster;
}

void CS397::FuzzyCMeans::Iteration()
{
	UpdateCentroids();
	auto values = Predict(m_data);
	ProbabilityMatrix = StoreProbabilityMatrix(values);
}

double CS397::FuzzyCMeans::Cost()
{
	double result = 0;
	for (int m = 0; m < m_column_samples; m++)
	{
		for (int k = 0; k < m_row_clusters; k++)
		{

			double w = ProbabilityMatrix[k * m_column_samples + m];

			std::vector<double> sample = m_data[m];

			double sample_cost = 0;
			for (int s = 0; s < sample.size(); s++)
			{
				double value = sample[s] - m_currentCentroids[k][s];
				sample_cost += value * value;
			}

			result += pow(w, m_fuzziness) * (sample_cost);

		}
	}

	return result / m_data.size();
}

double CS397::FuzzyCMeans::Cost(const Dataset & input)
{
	std::vector<std::vector<double>> values = Predict(input);
	//std::vector<double> matrix = StoreProbabilityMatrix(values);
	
	double result = 0;
	for (int m = 0; m < input.size(); m++)
	{
		for (int k = 0; k < m_row_clusters; k++)
		{
		
			double w = values[m][k];

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

	return result / input.size();
}

void CS397::FuzzyCMeans::UpdateCentroids()
{
	for (int k = 0; k < m_row_clusters; k++)
	{
		
		for (int s = 0; s < m_data[0].size(); s++)
		{
			double top_value = 0;
			double bot_value = 0;

			for (int i = 0; i < m_column_samples; i++)
			{
				double w = ProbabilityMatrix[m_column_samples * k + i];
				w = pow(w, m_fuzziness);

				top_value += w * m_data[i][s];
				bot_value += w;
			}

			m_currentCentroids[k][s] = top_value / bot_value;

		}

	}

}

std::vector<double> CS397::FuzzyCMeans::StoreProbabilityMatrix(const std::vector<std::vector<double>> & values)
{
	std::vector<double> matrix;
	for (int k = 0; k < m_row_clusters; k++)
	{
		for (int i = 0; i < m_column_samples; i++)
		{
			matrix.push_back(values[i][k]);
		}
	}

	return matrix;
}
