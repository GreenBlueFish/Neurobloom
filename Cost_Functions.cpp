#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <sstream>

#include "Cost_Functions.h"

double Cost::Difference_Squared(double correct_input, double actual_input) {
	return (0.5 * pow((actual_input - correct_input), 2));
}

double Cost::DERIVATIVE_Difference_Squared(double correct_input, double actual_input) {
	return (actual_input - correct_input);
}