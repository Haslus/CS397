//---------------------------------------------------------------------------
#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H
//---------------------------------------------------------------------------

#include <cmath>

namespace ActivationFunction
{

enum class Type
{
    eSigmoid,
    eTanh
};
double Sigmoid(double x);
double Tanh(double x);

double SigmoidDerivative(double x);
double TanhDerivative(double x);

} // namespace ActivationFunction

#endif