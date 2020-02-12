
#include "DatasetCreator.h"

namespace DatasetCreator
{
double SCALE = 5.0;

SplittedDataset GenerateBlobsDataset(unsigned sizeTraining, unsigned sizeTesting, unsigned numParams, unsigned numBlobs, double noise)
{
    SplittedDataset                  dataset({ Dataset(sizeTraining), Dataset(sizeTesting) });
    std::vector<std::vector<double>> blobCenters(numBlobs);
    const double                     NoiseSq = noise * noise;

    // generate blob centers
    for (unsigned b = 0; b < numBlobs; b++)
    {
        std::vector<double> center(numParams);
        for (unsigned j = 0; j < numParams; j++)
        {
            center[j] = (rand() * 1.0 / RAND_MAX) * SCALE;
        }

        blobCenters[b] = center;
    }

	// generate training dataset
    for (unsigned i = 0; i < sizeTraining; i++)
    {
        // generate random inputs
        std::vector<double> inputs(numParams);
        double              distanceSq = NoiseSq;
        unsigned            blobIndex  = rand() % numBlobs;

        while (distanceSq >= NoiseSq)
        {
            distanceSq = 0.0;
            for (unsigned j = 0; j < numParams; j++)
            {
                double noiseFactor = noise * (rand() * 2.0 / RAND_MAX - 1.0);
                inputs[j]          = noiseFactor + blobCenters[blobIndex][j];

                distanceSq += noiseFactor * noiseFactor;
            }
        }

        dataset.first[i] = inputs;
    }

    // generate testing dataset
    for (unsigned i = 0; i < sizeTesting; i++)
    {
        // generate random inputs
        std::vector<double> inputs(numParams);
        double              distanceSq = NoiseSq;
        unsigned            blobIndex  = rand() % numBlobs;

        while (distanceSq >= NoiseSq)
        {
            distanceSq = 0.0;
            for (unsigned j = 0; j < numParams; j++)
            {
                double noiseFactor = noise * (rand() * 2.0 / RAND_MAX - 1.0);
                inputs[j]          = noiseFactor + blobCenters[blobIndex][j];

                distanceSq += noiseFactor * noiseFactor;
            }
        }

        dataset.second[i] = inputs;
    }

    return dataset;
}
} // namespace DatasetCreator