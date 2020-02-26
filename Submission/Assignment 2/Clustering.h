//---------------------------------------------------------------------------
#ifndef CLUSTERING_H
#define CLUSTERING_H
//---------------------------------------------------------------------------

#include "DatasetCreator.h"

namespace CS397
{
class KMeans
{
  public:
    // Constructor
    KMeans(const Dataset &                          data,             // Dataset clusters will generated for
           const std::vector<std::vector<double>> & initialCentroids, // Initial centroid positions (number of clusters to generate is extracted from size)
           bool                                     meanNormalization);  // True if input data should be mean normalized

    // Given a single or multiple input datapoint(s), computes the index/indices
	// of the cluster each of the inputs belongs to
    std::vector<unsigned> Predict(const Dataset & input) const;
    unsigned              Predict(const std::vector<double> & input) const;

    // clusters will be adjusted for the dataset
    void Iteration();

    // Computes the cost for the training dataset
    double Cost();
    // Computes the cost of an external dataset
    double Cost(const Dataset & input);

	/*************************************************/
	//My Functions
	void CalculateClusters();
	void OutputClusters(Dataset dataset);
	std::vector<unsigned> Normalized_Predict() const;

	//My Parameters
	Dataset m_data;
	std::vector<std::vector<double>> m_initialCentroids;
	std::vector<std::vector<double>> m_currentCentroids;
	bool m_meanNormalization;
	struct NormalizedValues
	{
		NormalizedValues(double del, double m) : delta(del),mean(m) {};
		double delta, mean;
	};

	std::vector<NormalizedValues> m_norm_values;

	std::vector<unsigned> m_cluster_index;
	unsigned int m_number_of_clusters;

    
};

class FuzzyCMeans
{
private:
	static std::vector<double> InitialProbabilityMatrix(unsigned r, unsigned c)
	{
		std::vector<double> mat(r * c);

		// initialize probability matrix to random probabilities
		for (unsigned i = 0; i < r; i++)
		{
			double total = 0.0;

			// give random probabilities to each cluster
			for (unsigned j = 0; j < c; j++)
			{
				double value = static_cast<double>(std::rand()) / RAND_MAX;

				mat[i * c + j] = value;

				total += value;
			}

			// since all probabilities need to add up to 1 normalize probabilities
			for (unsigned j = 0; j < c; j++)
			{
				mat[i * c + j] /= total;
			}
		}

		return mat;
	}
    
  public:
    FuzzyCMeans(const Dataset &                          data,             // Dataset clusters will generated for
                const std::vector<std::vector<double>> & initialCentroids, // Initial centroid positions (number of clusters to generate is extracted from size)
                double                                   fuzziness,        // Specifies the fuzziness applied when learning
                bool                                     meanNormalization); // True if input data should be mean normalized

    // Given a single or multiple input datapoint(s), computes the probabilities of 
	// belonging to a cluster for each of the inputs
    std::vector<std::vector<double>> Predict(const Dataset & input) const;
    std::vector<double>              Predict(const std::vector<double> & input) const;

    // clusters will be adjusted for the dataset
    void Iteration();

    // Computes the cost for the training dataset
    double Cost();
    // Computes the cost of an external dataset
    double Cost(const Dataset & input);

	/*************************************************/
	//My Functions
	void UpdateCentroids();
	std::vector<double> StoreProbabilityMatrix(const std::vector<std::vector<double>> & values);
	std::vector<std::vector<double>> Normalized_Predict() const;

	//My Parameters
	Dataset m_data;
	double m_fuzziness;
	std::vector<std::vector<double>> m_initialCentroids;
	std::vector<std::vector<double>> m_currentCentroids;
	bool m_meanNormalization;
	struct NormalizedValues
	{
		NormalizedValues(double del, double m) : delta(del), mean(m) {};
		double delta, mean;
	};

	std::vector<NormalizedValues> m_norm_values;

	unsigned int m_row_clusters;
	unsigned int m_column_samples;
	std::vector<double> ProbabilityMatrix;
};
} // namespace CS397

#endif