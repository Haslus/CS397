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
	//My Parameters
	void CalculateClusters();

	Dataset m_data;
	std::vector<std::vector<double>> m_initialCentroids;
	std::vector<std::vector<double>> m_currentCentroids;
	bool m_meanNormalization;
	std::vector<unsigned> m_cluster_index;
	int m_number_of_clusters;

    
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

};
} // namespace CS397

namespace CS397_Student
{
	struct Link
	{
		int m_sample_index;
		int m_centroid_index;
	};
}

#endif