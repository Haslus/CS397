
#include "gtest/gtest.h"

#include "Clustering.h"
#include "DatasetCreator.h"

#include <fstream>

using namespace CS397;
using namespace DatasetCreator;

template <typename T>
void AssertVectors(const std::vector<T> & val1, const std::vector<T> & val2)
{
    ASSERT_EQ(val1.size(), val2.size());

    for (size_t i = 0; i < val1.size(); i++)
    {
        ASSERT_NEAR(val1[i], val2[i], 0.01);
    }
}

// K means
TEST(KMeans, Cost_1D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 2;
    const unsigned FeatureCount = 1;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3] }, false);

    double cost = kMeans.Cost(trainSet);

    ASSERT_NEAR(cost, 0.17, 0.01);

    double testCost = kMeans.Cost(testSet);

    ASSERT_NEAR(testCost, 0.17, 0.01);
}

TEST(KMeans, Cost_2D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 20;
    const unsigned FeatureCount = 2;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3] }, false);

    double cost = kMeans.Cost(trainSet);

    ASSERT_NEAR(cost, 2.80, 0.01);

    double testCost = kMeans.Cost(testSet);

    ASSERT_NEAR(testCost, 2.80, 0.01);
}

TEST(KMeans, Cost_10D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 20;
    const unsigned FeatureCount = 10;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3] }, false);

    double cost = kMeans.Cost(trainSet);

    ASSERT_NEAR(cost, 22.13, 0.01);

    double testCost = kMeans.Cost(testSet);

    ASSERT_NEAR(testCost, 23.35, 0.01);
}

TEST(KMeans, Iterate1D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 3;
    const unsigned FeatureCount = 1;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2] }, false);

    double cost = kMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.60, 0.01);

    for (unsigned i = 0; i < 2; i++)
    {
        kMeans.Iteration();
    }

    cost = kMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.26, 0.01);
}

TEST(KMeans, Iterate2D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 5;
    const unsigned FeatureCount = 2;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2] }, false);

    double cost = kMeans.Cost(testSet);

    ASSERT_NEAR(cost, 1.56, 0.01);

    for (unsigned i = 0; i < 5; i++)
    {
        kMeans.Iteration();
    }

    cost = kMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.77, 0.01);
}

TEST(KMeans, Iterate10D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 5;
    const unsigned FeatureCount = 10;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3], trainSet[4] }, false);

    double cost = kMeans.Cost(testSet);

    ASSERT_NEAR(cost, 7.15, 0.01);

    for (unsigned i = 0; i < 5; i++)
    {
        kMeans.Iteration();
    }

    cost = kMeans.Cost(testSet);

    ASSERT_NEAR(cost, 4.81, 0.01);
}

TEST(KMeans, Predict2D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 5;
    const unsigned FeatureCount = 2;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 1, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1] }, false);

    for (unsigned i = 0; i < 10; i++)
    {
        kMeans.Iteration();
    }

    unsigned prediction = kMeans.Predict(testSet[0]);

    ASSERT_EQ(prediction, 1);
}

TEST(KMeans, Predict5D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 3;
    const unsigned FeatureCount = 5;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 1, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    KMeans kMeans = KMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2] }, false);

    for (unsigned i = 0; i < 10; i++)
    {
        kMeans.Iteration();
    }

    unsigned prediction = kMeans.Predict(testSet[0]);

    ASSERT_EQ(prediction, 1);
}

// Fuzzy C means
TEST(FuzzyCMeans, Cost_1D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 2;
    const unsigned FeatureCount = 1;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 2.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3] }, Fuzziness, false);

    double cost = fuzzyCMeans.Cost(trainSet);

    ASSERT_NEAR(cost, 0.08, 0.01);

    cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.08, 0.01);
}
				  
TEST(FuzzyCMeans, Cost_2D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 20;
    const unsigned FeatureCount = 2;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 2.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3] }, Fuzziness, false);

    double cost = fuzzyCMeans.Cost(trainSet);

    ASSERT_NEAR(cost, 1.29, 0.01);

    cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 1.33, 0.01);
}
				  
TEST(FuzzyCMeans, Cost_10D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 20;
    const unsigned FeatureCount = 10;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 5.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3] }, Fuzziness, false);

    double cost = fuzzyCMeans.Cost(trainSet);

    ASSERT_NEAR(cost, 0.14, 0.01);

    cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.14, 0.01);
}
				  
TEST(FuzzyCMeans, Iterate1D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 3;
    const unsigned FeatureCount = 1;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness = 2.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2] }, Fuzziness, false);

    double cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.34, 0.01);

    for (unsigned i = 0; i < 2; i++)
    {
        fuzzyCMeans.Iteration();
    }

    cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 0.8, 0.01);
}
				  
TEST(FuzzyCMeans, Iterate2D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 5;
    const unsigned FeatureCount = 2;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 2.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2] }, Fuzziness, false);

    double cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 1.03, 0.01);

    for (unsigned i = 0; i < 20; i++)
    {
        fuzzyCMeans.Iteration();
		cost = fuzzyCMeans.Cost(testSet);
    }

    cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 1.23, 0.01);
}
				  
TEST(FuzzyCMeans, Iterate10D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 5;
    const unsigned FeatureCount = 10;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 100, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 2.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2], trainSet[3], trainSet[4] }, Fuzziness, false);

    double cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 2.91, 0.01);

    for (unsigned i = 0; i < 5; i++)
    {
        fuzzyCMeans.Iteration();
    }

    cost = fuzzyCMeans.Cost(testSet);

    ASSERT_NEAR(cost, 4.64, 0.01);
}

TEST(FuzzyCMeans, Predict2D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 5;
    const unsigned FeatureCount = 2;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 1, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 5.0;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1] }, Fuzziness, false);

    for (unsigned i = 0; i < 10; i++)
    {
        fuzzyCMeans.Iteration();
    }

    std::vector<double> prediction = fuzzyCMeans.Predict(testSet[0]);

	AssertVectors(prediction, { 0.38, 0.62 });	
}

TEST(FuzzyCMeans, Predict5D)
{
    const double   BlobSize     = 1.0;
    const unsigned NumClusters  = 3;
    const unsigned FeatureCount = 5;

    srand(0);
    SplittedDataset splittedDataset = GenerateBlobsDataset(1000, 1, FeatureCount, NumClusters, BlobSize);

    const Dataset & trainSet = splittedDataset.first;
    const Dataset & testSet  = splittedDataset.second;

    const double Fuzziness   = 1.5;
    FuzzyCMeans  fuzzyCMeans = FuzzyCMeans(trainSet, { trainSet[0], trainSet[1], trainSet[2] }, Fuzziness, false);

    for (unsigned i = 0; i < 10; i++)
    {
        fuzzyCMeans.Iteration();
    }

    std::vector<double> prediction = fuzzyCMeans.Predict(testSet[0]);

    AssertVectors(prediction, { 0.99, 0.01, 0.00 });
}

// Student tests
Dataset LoadIris()
{
    Dataset dataset;
    // read real state dataset
    std::ifstream infile;
    infile.open("Iris.csv", std::ifstream::in);
    if (infile.is_open())
    {
        char numStr[256];
        // skip first line with headers
        infile.getline(numStr, 256);
        while (infile.good())
        {
            // ignore first element (just the index)
            infile.getline(numStr, 256, ',');

            // no element to read on this line so skip
            if (infile.eof())
                continue;

            // read datapoint
            std::vector<double> input;

            // each datapoint contains 4 attributes
            for (unsigned i = 0; i < 4; i++)
            {
                infile.getline(numStr, 256, ',');
                input.push_back(std::atof(numStr));
            }
            // skip class
            infile.getline(numStr, 256, '\n');

            dataset.push_back(input);
        }
        infile.close();
    }

    return dataset;
}

TEST(KMeans, Iris)
{
    Dataset dataset = LoadIris();

	int size = 75;
	Dataset trainSet = { dataset.begin(), dataset.begin() + size };
	Dataset testSet = { dataset.begin() + size, dataset.end() };
    // STUDENT TEST
	const double   BlobSize = 1.0;
	const unsigned NumClusters = 3;
	const unsigned FeatureCount = 4;
	srand(1);
	int rand1 = rand() % size;
	int rand2 = rand() % size;
	int rand3 = rand() % size;
	KMeans kMeans = KMeans(trainSet, { trainSet[rand1], trainSet[rand2], trainSet[rand3] }, false);

	double cost = kMeans.Cost(testSet);

	for (unsigned i = 0; i < 3; i++)
	{
		kMeans.Iteration();
		cost = kMeans.Cost(testSet);
	}

	kMeans.OutputClusters();
}

Dataset LoadMallCustomers()
{
    Dataset dataset;
    // read real state dataset
    std::ifstream infile;
    infile.open("Mall_Customers.csv", std::ifstream::in);
    if (infile.is_open())
    {
        char numStr[256];
        // skip first line with headers
        infile.getline(numStr, 256);
        while (infile.good())
        {
            // ignore first element (just the index)
            infile.getline(numStr, 256, ',');

            // no element to read on this line so skip
            if (infile.eof())
                continue;

            // read datapoint
            std::vector<double> input;

            // each datapoint contains 9 attributes
            for (unsigned i = 0; i < 3; i++)
            {
                infile.getline(numStr, 256, ',');
                input.push_back(std::atof(numStr));
            }
            infile.getline(numStr, 256, '\n');
            input.push_back(std::atof(numStr));

            dataset.push_back(input);
        }
        infile.close();
    }

    return dataset;
}

TEST(KMeans, MallCustomers)
{
    Dataset dataset = LoadMallCustomers();

    // STUDENT TEST
}

int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);

    int toReturn = RUN_ALL_TESTS();

    return toReturn;
}