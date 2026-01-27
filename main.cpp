#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <sstream>

//Main file
#include "Core.h"

//Macro files
#include "C:\Users\rowan\Documents\Coding_Projects\Joels_Code\MacroTools.h"

#include "Main_Network_Support_Files\Activation_Functions.h"

int main()
{
	Neural_Network test;

	std::cout << "Act Func Derivative 1: " << Activation::DERIVATIVE_Sigmoid(2.174502183) << std::endl;
	std::cout << "Act Func Derivative 2: " << Activation::DERIVATIVE_TanH(2.174502183) << std::endl;
	std::cout << "Act Func Derivative 3: " << Activation::DERIVATIVE_ReLU(2.174502183) << std::endl;
	std::cout << "Act Func Derivative 4: " << Activation::DERIVATIVE_Leaky_ReLU(2.174502183) << std::endl;
	std::cout << "Act Func Derivative 5: " << Activation::DERIVATIVE_Sine(2.174502183) << std::endl;
	std::cout << "Act Func Derivative 6: " << Activation::DERIVATIVE_Cosine(2.174502183) << std::endl;

	test.Print_All_D_Acts(2.174502183);

	std::vector<unsigned> test_act_func_input(2, 0);
	std::vector<unsigned> test_act_func_back(2, 1);

	test.Setting_No_Rand_Bias = 3;
	test.Setting_No_Rand_Weights = 5;

	test.Add_Layer(2, &test_act_func_input);
	test.Add_Layer(2, &test_act_func_back);
	test.Add_Layer(2, &test_act_func_back);

	//test.Add_Cost_Function(1, 1, 0);
	//test.Remove_Cost_Function(1, 1);

	test.Add_Connection(0, 0, 2, 1);

	test.Rand_High_Range = 2.5;
	test.Rand_Low_Range = -2.5;
	test.Rand_Decimal_Multiplier = 1000;

	test.Randomize_weight_values();
	test.Randomize_bias_values();

	test.Learning_Rate = 0.001;

	std::vector<double> output(2, 4);
	output[1] = -3;

	test.Print_Whole_Network();
	std::cout << std::endl << std::endl;
	test.Print_Cost_Nodes(&output);
	std::cout << std::endl << std::endl;
	std::cout << "COST: " << test.Network_Cost(&output) << std::endl << std::endl << std::endl;

	std::vector<double> input(2, -2);
	input[1] = 3;

	test.Forward_Propogate(&input);

	test.Print_Whole_Network();
	std::cout << std::endl << std::endl;
	test.Print_Cost_Nodes(&output);
	std::cout << std::endl << std::endl;
	std::cout << "COST: " << test.Network_Cost(&output) << std::endl << std::endl << std::endl;

	test.Back_Propogate_Gradient(&output);

	test.Print_Whole_Network();
	std::cout << std::endl << std::endl;
	test.Print_Cost_Nodes(&output);
	std::cout << std::endl << std::endl;
	std::cout << "COST: " << test.Network_Cost(&output) << std::endl << std::endl << std::endl;

	//std::vector<unsigned> test_act_func_new(3, 0);
	//test.Add_Layer(9, 3, &test_act_func_new);

	//test.Add_Connection(0, 0, 2, 1);

	//test.Print_Whole_Network();
}