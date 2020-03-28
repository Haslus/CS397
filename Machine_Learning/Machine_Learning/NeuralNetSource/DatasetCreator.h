//---------------------------------------------------------------------------
#ifndef DATASETCREATOR_H
#define DATASETCREATOR_H
//---------------------------------------------------------------------------

#include <vector>

namespace DatasetCreator
{

const double SCALE = 1.0;
const double NOISE = 0.0;

// the pair contains inputs and outputs of a dataset
// first is just a list of inputs (each input may have multiple values, hence the vector)
// second is the output for each of those samples (each output may contain multiple values, hence the vector)
using Dataset = std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>;

Dataset GenerateXDataset(unsigned size);
Dataset GenerateQuadrantsDataset(unsigned size);
Dataset GenerateRingDataset(unsigned size);
Dataset GenerateSineDataset(unsigned size);
Dataset GenerateCrossDataset(unsigned size);

Dataset GenerateColorQuadrantsDataset(unsigned size);
Dataset GenerateColorRingDataset(unsigned size);
Dataset GenerateColorSpiralDataset(unsigned size);

} // namespace DatasetCreator

#endif