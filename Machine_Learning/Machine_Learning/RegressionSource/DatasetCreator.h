//---------------------------------------------------------------------------
#ifndef DATASET_CREATION_H
#define DATASET_CREATION_H
//---------------------------------------------------------------------------

#include <vector>


// Dataset type contiains a pair where first represents the input data
// and the second stores all the outputs (both vectors should always same size)<
using Dataset = std::pair<std::vector<std::vector<double>>, std::vector<double>>;

namespace DatasetCreator
{

struct FunctionFeature
{
    int    index;
    int    power;
    double weight;
};

Dataset GenerateDataset(unsigned size, std::vector<FunctionFeature> & features, int numberInputs, double scale, double noise);

} // namespace DatasetCreator

#endif