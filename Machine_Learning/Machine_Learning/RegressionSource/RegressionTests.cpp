
#include "gtest/gtest.h"

#include "DatasetCreator.h"
#include "Regression.h"

#include <fstream>
#include <vector>

using namespace DatasetCreator;
using namespace CS397;

template <typename T>
void AssertVectors(const std::vector<T> & val1, const std::vector<T> & val2)
{
    ASSERT_EQ(val1.size(), val2.size());

    for (size_t i = 0; i < val1.size(); i++)
    {
        ASSERT_NEAR(val1[i], val2[i], 0.01);
    }
}

TEST(Regression, PredictLinear)
{
    double                       learningRate     = 0.01;
    double                       scale            = 50.0;
    double                       noise            = 2.0;
    unsigned                     numberInputs     = 1;
    std::vector<FunctionFeature> functionFeatures = { { 0, 1, 0.4 } };

    srand(0);
    Dataset dataset = GenerateDataset(5, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { 0, 1, 0.2 } };
    Regression           regression = Regression(dataset, features, learningRate, false);

    // compute prediction
    std::vector<double> output = regression.Predict(dataset.first);

    AssertVectors<double>({ 0.01, 6.48, 2.70, 2.55, 3.19 }, output);
}

TEST(Regression, PredictLinearWithIntercept)
{
    double                       learningRate     = 0.01;
    double                       scale            = 50.0;
    double                       noise            = 2.0;
    unsigned                     numberInputs     = 1;
    std::vector<FunctionFeature> functionFeatures = { { 0, 1, 0.4 }, { -1, 0, 0.2 } };

    srand(0);
    Dataset dataset = GenerateDataset(5, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { 0, 1, 0.2 }, { -1, 0, 0.4 } };
    Regression           regression = Regression(dataset, features, learningRate, false);

    // compute prediction
    std::vector<double> output = regression.Predict(dataset.first);

    AssertVectors<double>({ 0.41, 6.88, 3.10, 2.95, 3.59 }, output);
}

TEST(Regression, PredictSquareWithIntercept)
{
    double                       learningRate     = 0.01;
    double                       scale            = 50.0;
    double                       noise            = 2.0;
    unsigned                     numberInputs     = 1;
    std::vector<FunctionFeature> functionFeatures = { { 0, 2, 0.4 }, { -1, 0, 0.2 } };

    srand(0);
    Dataset dataset = GenerateDataset(5, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { 0, 2, 0.3 }, { -1, 0, 0.4 } };
    Regression           regression = Regression(dataset, features, learningRate, false);

    // compute prediction
    std::vector<double> output = regression.Predict(dataset.first);

    AssertVectors<double>({ 0.40, 315.48, 55.17, 49.28, 76.68 }, output);
}

TEST(Regression, LearnLinearWithIntercept)
{
    double                       learningRate     = 0.01;
    double                       scale            = 1.0;
    double                       noise            = 0.1;
    unsigned                     numberInputs     = 1;
    std::vector<FunctionFeature> functionFeatures = { { -1, 0, 0.5 }, { 0, 1, -0.5 } };

    srand(0);
    Dataset dataset  = GenerateDataset(1000, functionFeatures, numberInputs, scale, noise);
    Dataset test_set = GenerateDataset(100, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { -1, 0, 0.2 }, { 0, 1, 0.0 } };
    Regression           regression = Regression(dataset, features, learningRate, false);

    // train regression
    for (size_t i = 0; i < 1000; i++)
    {
        regression.Iteration();
    }

    // compute prediction
    std::vector<double> output = regression.Predict(test_set.first);

    double cost = regression.Cost(output, test_set.second);

    ASSERT_LT(cost, 0.005);
}

TEST(Regression, LearnSquareWithIntercept)
{
    double                       learningRate     = 0.01;
    double                       scale            = 1.0;
    double                       noise            = 0.01;
    unsigned                     numberInputs     = 1;
    std::vector<FunctionFeature> functionFeatures = { { -1, 0, 0.5 }, { 0, 1, -0.5 }, { 0, 2, 0.6 } };

    srand(0);
    Dataset dataset  = GenerateDataset(1000, functionFeatures, numberInputs, scale, noise);
    Dataset test_set = GenerateDataset(100, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { -1, 0, 0.2 }, { 0, 1, -0.6 }, { 0, 2, 0.4 } };
    Regression           regression = Regression(dataset, features, learningRate, false);

    // train regression
    for (size_t i = 0; i < 1000; i++)
    {
        regression.Iteration();
    }

    // compute prediction
    std::vector<double> output = regression.Predict(test_set.first);

    double cost = regression.Cost(output, test_set.second);

    ASSERT_LT(cost, 0.005);
}

TEST(Regression, LearnMultivariableLinearWithIntercept)
{
    double                       learningRate     = 0.01;
    double                       scale            = 1.0;
    double                       noise            = 0.1;
    unsigned                     numberInputs     = 2;
    std::vector<FunctionFeature> functionFeatures = { { -1, 0, 0.5 }, { 0, 1, -0.5 }, { 1, 1, 0.6 } };

    srand(0);
    Dataset dataset  = GenerateDataset(1000, functionFeatures, numberInputs, scale, noise);
    Dataset test_set = GenerateDataset(100, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { -1, 0, 0.2 }, { 0, 1, -0.3 }, { 1, 1, 0.4 } };
    Regression           regression = Regression(dataset, features, learningRate, false);

    // train regression
    for (size_t i = 0; i < 1000; i++)
    {
        regression.Iteration();
    }

    // compute prediction
    std::vector<double> output = regression.Predict(test_set.first);

    double cost = regression.Cost(output, test_set.second);

    ASSERT_LT(cost, 0.01);
}

TEST(Regression, LearnMultivariableWithIntercept)
{
    double                       learningRate     = 0.01;
    double                       scale            = 1.0;
    double                       noise            = 0.1;
    unsigned                     numberInputs     = 3;
    std::vector<FunctionFeature> functionFeatures = { { -1, 0, 0.5 }, { 0, 1, -0.5 }, { 1, 2, 0.1 }, { 2, 3, 0.2 } };

    srand(0);
    Dataset dataset  = GenerateDataset(1000, functionFeatures, numberInputs, scale, noise);
    Dataset test_set = GenerateDataset(100, functionFeatures, numberInputs, scale, noise);

    std::vector<Feature> features   = { { -1, 0, 0.0 }, { 0, 1, 0.0 }, { 1, 2, 0.0 }, { 2, 3, 0.0 } };
    Regression           regression = Regression(dataset, features, learningRate, true);

    // train regression
    for (size_t i = 0; i < 1000; i++)
    {
        regression.Iteration();
    }

    // compute prediction
    std::vector<double> output = regression.Predict(test_set.first);

    double cost = regression.Cost(output, test_set.second);

    ASSERT_LT(cost, 0.01);
}

Dataset LoadRealStateCSV()
{
    Dataset dataset;
    // read real state dataset
    std::ifstream infile;
    infile.open("RealEstatePrices.csv", std::ifstream::in);
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
            double              output = 0.0;

            // each datapoint contains 6 attributes
            for (unsigned i = 0; i < 6; i++)
            {
                infile.getline(numStr, 256, ',');
                input.push_back(std::atof(numStr));
            }
            infile.getline(numStr, 256);
            output = std::atof(numStr);

            dataset.first.push_back(input);
            dataset.second.push_back(output);
        }
        infile.close();
    }

    return dataset;
}

TEST(Regression, HousePriceDeduction)
{
    double   learningRate = 0.01;
    unsigned numberInputs = 6;
    Dataset  dataset      = LoadRealStateCSV();

    // STUDENT TEST


    ASSERT_LT(0.0, 45.0);
}
int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);

    int toReturn = RUN_ALL_TESTS();

    return toReturn;
}