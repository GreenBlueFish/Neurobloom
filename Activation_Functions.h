#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>

namespace Activation
{
    //Activation functions and their derivatives
    double Null(double Input);
    double DERIVATIVE_Null(double Input);

    double Sigmoid(double Input);
    double DERIVATIVE_Sigmoid(double Input);

    double TanH(double Input);
    double DERIVATIVE_TanH(double Input);

    double ReLU(double Input);
    double DERIVATIVE_ReLU(double Input);

    double Leaky_ReLU(double Input);
    double DERIVATIVE_Leaky_ReLU(double Input);

    double Sine(double Input);
    double DERIVATIVE_Sine(double Input);

    double Cosine(double Input);
    double DERIVATIVE_Cosine(double Input);
}
#endif