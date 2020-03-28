#include "ActivationFunctions.h"

namespace ActivationFunction
{

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Tanh(double x) { return std::tanh(x); }

double SigmoidDerivative(double sigmoid)
{
    return (1.0 - sigmoid) * sigmoid;
}
double TanhDerivative(double tanh)
{
    return 1.0 - tanh * tanh;
}
} // namespace ActivationFunction