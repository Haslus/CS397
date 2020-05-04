
#include "mnist/mnist_reader.hpp"
#include "gtest/gtest.h"

#include "DatasetCreator.h"
#include "NeuralNet.h"

#include <fstream>

using namespace DatasetCreator;
using namespace CS397;

TEST(NeuralNet, RandomWeightInit)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 3, 1 });
    const double                LearningRate = 0.01;

    NeuralNet net = NeuralNet(Dataset(), Topology, LearningRate);

    const NetworkWeights result = {
        { { -0.998, -0.529, 0.296 }, { -0.851, -0.460, -0.280 }, { -0.489,0.971,-0.362 } },
        { { 0.868 }, { -0.643 }, { 0.715 }, { -0.930 } },
        { {} }
    };

    // retrieve weights and compare
    NetworkWeights weigths = net.GetWeights();

    for (size_t l = 0; l < result.size(); l++)
    {
        for (size_t n = 0; n < result[l].size(); n++)
        {
            for (size_t w = 0; w < result[l][n].size(); w++)
            {
                EXPECT_NEAR(weigths[l][n][w], result[l][n][w], 0.001);
            }
        }
    }
}

TEST(NeuralNet, ForwardPropagation_2_1)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 3, 1 });
    const double                LearningRate = 0.01;

    NeuralNet net = NeuralNet(Dataset(), Topology, LearningRate);

    // process input with the initial random weights
    std::vector<double> input { 2.0, 0.5 };
    std::vector<double> output = net.ForwardPropagation(input);

    // check output size and values
    EXPECT_EQ(output.size(), 1);
    EXPECT_NEAR(output[0], 0.314, 0.001);
}

TEST(NeuralNet, ForwardPropagation_2_3)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 3, 3 });
    const double                LearningRate = 0.01;

    NeuralNet net = NeuralNet(Dataset(), Topology, LearningRate);

    // process input with the initial random weights
    std::vector<double> input { 2.0, 0.5 };
    std::vector<double> output = net.ForwardPropagation(input);

    // check output size and values
    const std::vector<double> result = { 0.235, 0.312, 0.677 };
    EXPECT_EQ(output.size(), 3);
    for (size_t i = 0; i < output.size(); i++)
    {
        EXPECT_NEAR(output[i], result[i], 0.001);
    }
}

TEST(NeuralNet, Cost_2_1)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 2, 1 });
    const double                LearningRate = 0.01;

    // generate datasets
    Dataset train_dataset = DatasetCreator::GenerateXDataset(200);
    Dataset test_dataset  = DatasetCreator::GenerateXDataset(30);

    NeuralNet net = NeuralNet(train_dataset, Topology, LearningRate);

    // test dataset to compute average error
    double cost = net.Cost(test_dataset);

    // check cost value
    EXPECT_NEAR(cost, 0.251, 0.001);
}

TEST(NeuralNet, IterationBPXDataset_2_1)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 2, 1 });
    const double                LearningRate = 0.01;
    const unsigned              Epochs       = 1;

    // generate datasets
    Dataset train_dataset = DatasetCreator::GenerateXDataset(200);
    Dataset test_dataset  = DatasetCreator::GenerateXDataset(30);

    NeuralNet net = NeuralNet(train_dataset, Topology, LearningRate, ActivationFunction::Type::eTanh);
	
    // train dataset multiple times
    for (size_t e = 0; e < Epochs; e++)
    {
        net.Iteration();
    }

    const NetworkWeights resultPostIteration = {
        { { -0.081607, 0.216113 }, { 0.841705, -0.402012 }, { -0.496121, -0.259302 } },
        { { -0.236406 }, { 0.504009 }, { -0.021947 } },
        { {} }
    };
	
    // retrieve weights and compare
    NetworkWeights weigths = net.GetWeights();

    for (size_t l = 0; l < resultPostIteration.size(); l++)
    {
        for (size_t n = 0; n < resultPostIteration[l].size(); n++)
        {
            for (size_t w = 0; w < resultPostIteration[l][n].size(); w++)
            {
                EXPECT_NEAR(weigths[l][n][w], resultPostIteration[l][n][w], 0.001);
            }
        }
    }

    // test dataset to compute average error
    double cost = net.Cost(test_dataset);

    // check cost value
    EXPECT_LT(cost, 0.2);
}

TEST(NeuralNet, IterationXDataset_2_1)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 2, 1 });
    const double                LearningRate = 0.01;
    const unsigned              Epochs       = 500;

    // generate datasets
    Dataset train_dataset = DatasetCreator::GenerateXDataset(200);
    Dataset test_dataset  = DatasetCreator::GenerateXDataset(30);

    NeuralNet net = NeuralNet(train_dataset, Topology, LearningRate, ActivationFunction::Type::eTanh);

    // train dataset multiple times
    for (size_t e = 0; e < Epochs; e++)
    {
        net.Iteration();
    }

    // test dataset to compute average error
    double cost = net.Cost(test_dataset);

    // check cost value
    EXPECT_LT(cost, 0.1);
}

TEST(NeuralNet, IterationRingDataset_10_8_3)
{
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 2, 10, 8, 3 });
    const double                LearningRate = 0.01;
    const unsigned              Epochs       = 500;

    // generate datasets
    Dataset train_dataset = DatasetCreator::GenerateColorRingDataset(1000);
    Dataset test_dataset  = DatasetCreator::GenerateColorRingDataset(100);

    NeuralNet net = NeuralNet(train_dataset, Topology, LearningRate);

    // train dataset multiple times
    for (size_t e = 0; e < Epochs; e++)
    {
        net.Iteration();
    }

    // test dataset to compute average error
    double cost = net.Cost(test_dataset);

    // check cost value
    EXPECT_LT(cost, 0.2);
}

std::pair<Dataset, Dataset> LoadHandwrittenDigits()
{
    std::pair<Dataset, Dataset> splittedDataset;
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("handwritten/");

    Dataset & training_dataset = splittedDataset.first;
    Dataset & test_dataset     = splittedDataset.second;

    const unsigned TrainingSize = unsigned(dataset.training_images.size());
    // copy mnist training data to local format
    training_dataset = Dataset({ TrainingSize, std::vector<double>(784, 0.0) },
                                               { TrainingSize, std::vector<double>(10, 0.0) });
    for (unsigned i = 0; i < TrainingSize; i++)
    {
        for (unsigned j = 0; j < dataset.training_images[i].size(); j++)
        {
            training_dataset.first[i][j] = dataset.training_images[i][j];
        }
        training_dataset.second[i][dataset.training_labels[i]] = 1.0;
    }

    // copy mnist test data to local format
    const unsigned TestingSize = unsigned(dataset.test_images.size());
    test_dataset               = Dataset({ TestingSize, std::vector<double>(784, 0.0) },
                                           { TestingSize, std::vector<double>(10, 0.0) });
    for (unsigned i = 0; i < TestingSize; i++)
    {
        for (unsigned j = 0; j < dataset.test_images[i].size(); j++)
        {
            test_dataset.first[i][j] = dataset.test_images[i][j];
        }
        test_dataset.second[i][dataset.test_labels[i]] = 1.0;
    }

    return splittedDataset;
}

double ComputeHandwrittenAccuracy(NeuralNet & net, Dataset & data)
{
    double accuracy = 0.0f;

    // compute the accuracy by measuring if the value was correctly classified
    for (size_t i = 0; i < data.first.size(); i++)
    {
        std::vector<double> output       = net.ForwardPropagation(data.first[i]);
        double              avgErrorTest = 0.0f;

        unsigned guessedNum    = 0;
        double   maxGuessValue = output[0];
        unsigned realNum       = 0;
        double   maxRealValue  = data.second[i][0];

        // get guessed number that has greatest value
        // and the real number in the dataset
        for (unsigned j = 1; j < output.size(); j++)
        {
            if (maxGuessValue < output[j])
            {
                guessedNum    = j;
                maxGuessValue = output[j];
            }
            if (maxRealValue < data.second[i][j])
            {
                realNum      = j;
                maxRealValue = output[j];
            }
        }
        if (realNum == guessedNum)
            accuracy += 1.0;

      //  std::cout << "P: " << guessedNum << " R:" << realNum << std::endl;
    }

    return accuracy / data.first.size();
}

TEST(NeuralNet, Handwritten)
{
    // STUDENT TEST
    
    srand(0);
    // net config
    const std::vector<unsigned> Topology({ 784, 100, 30, 10 });
    const double                LearningRate = 0.01;
    const unsigned              Epochs       = 5;

    std::pair<Dataset, Dataset> splittedDataset = LoadHandwrittenDigits();

    Dataset & training_dataset = splittedDataset.first;
    Dataset & test_dataset     = splittedDataset.second;

    // generate datasets
    NeuralNet net = NeuralNet(training_dataset, Topology, LearningRate);

    // train dataset multiple times
    for (size_t e = 0; e < Epochs; e++)
    {
        std::cout << "Cost: " << net.Cost(test_dataset) << std::endl;
        std::cout << "Accuracy: " << ComputeHandwrittenAccuracy(net, test_dataset) << std::endl;
        std::cout << ComputeHandwrittenAccuracy(net, test_dataset) << std::endl;

        net.Iteration();
    }

    double accuracy = ComputeHandwrittenAccuracy(net, test_dataset);

    // should at least accuratelly clasify 80% of the images
    EXPECT_GE(accuracy, 0.8);
}

int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);

    int toReturn = RUN_ALL_TESTS();

    std::system("pause");

    return toReturn;
}