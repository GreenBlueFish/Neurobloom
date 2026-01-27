#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <sstream>

namespace Cost
{
	double Difference_Squared(double correct_input, double actual_input);
	double DERIVATIVE_Difference_Squared(double correct_input, double actual_input);
}

#endif