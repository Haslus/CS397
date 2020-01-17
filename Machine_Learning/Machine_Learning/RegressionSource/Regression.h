//---------------------------------------------------------------------------
#ifndef REGRESSION_H
#define REGRESSION_H
//---------------------------------------------------------------------------

#include "DatasetCreator.h"

namespace CS397
{

struct Feature
{
    int    inputIdx; // Index of the input this feature refers to (negative index means intercept)
    int    power;    // Power of the feature
    double theta;    // Constant multiplier of the feature (what the algorithm will adjust)
};

struct FeatureNorm
{
	double mean;
	double range;
};

class Regression
{
  public:
    // Constructor
    Regression(const Dataset &              dataset,  // Dataset that the regression will learn from
               const std::vector<Feature> & features, // Features
               double                       lr,       // Learning rate fo the linear regression
               bool                         meanNormalization);               // True if input data should be mean normalized

    // Given a single or multiple input datapoint(s), computes the value(s) that the linear regression would predict
    double              Predict(const std::vector<double> & input) const;
    std::vector<double> Predict(const std::vector<std::vector<double>> & input) const;

    // Given the output predicted and the actual real value (target), computes the cost
    double Cost(const std::vector<double> & output,
                const std::vector<double> & target) const;

    // Linear regression values will be adjusted for the dataset using gradient descent
    void Iteration();

	//My Functions
	double Cost_Derivative(const std::vector<double>& output, const std::vector<double>& target);
  
private:
	  std::vector<Feature> features;
	  Dataset dataset;
	  float lr;
	  bool meanNormalization;
	  std::vector<FeatureNorm> feature_norms;
    
};

} // namespace CS397
#endif