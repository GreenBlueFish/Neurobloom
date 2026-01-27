#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <sstream>

#include "Activation_Functions.h"

// Activation functions and their derivatives
double Activation::Null(double Input) {
    return Input;
}
double Activation::DERIVATIVE_Null(double Input) {
    return 1;
}

double Activation::Sigmoid(double Input) {
    return 1 / (1 + pow(2.7182818, -Input));
}
double Activation::DERIVATIVE_Sigmoid(double Input) {
    return pow((pow(2.7182818, -Input) + 1), -2) * pow(2.7182818, -Input);
}

double Activation::TanH(double Input) {
    return (((2) / (1 + pow(2.7182818, -2 * Input))) - 1);
}
double Activation::DERIVATIVE_TanH(double Input) {
    return (1 - pow(TanH(Input), 2));
}

double Activation::ReLU(double Input) {
    if (Input <= 0) { return 0; }

    return Input;
}
double Activation::DERIVATIVE_ReLU(double Input) {
    if (Input <= 0) { return 0; }
    return 1;
}

double Activation::Leaky_ReLU(double Input) {
    if (Input >= 0) { return Input; }
    else { return 0.01 * Input; }
}
double Activation::DERIVATIVE_Leaky_ReLU(double Input) {
    if (Input >= 0) { return 1; }
    else { return 0.01; }
}

double Activation::Sine(double Input) {
    return sin(Input);
}
double Activation::DERIVATIVE_Sine(double Input) {
    return cos(Input);
}

double Activation::Cosine(double Input) {
    return cos(Input);
}
double Activation::DERIVATIVE_Cosine(double Input) {
    return (-1 * sin(Input));
}