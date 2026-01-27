#ifndef CORE_H
#define CORE_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>

class Neural_Network
{
private:
	struct extra_connection;

	struct node {
		unsigned Act_func = 0;
		long double value_act = 0;
		long double value_un_act = 0;

		long double bias = 0;

		std::vector<extra_connection> backward_extra;

		long double back_prop_data = 0;

		bool has_cost = true;
		unsigned Cost_func = 0;

	};
	struct extra_connection {
		node* node_connection;
		long double weight = 1;
	};
	struct layer {
		std::vector<node> layer_nodes;

		//weight[current layer][previous layer]
		std::vector<std::vector<double>> weights;
	};

	//For vector of activation function
	using func_ptr_acivation = double(*)(double);

	std::vector<func_ptr_acivation> Activation_Function_Vect;       //Holds all pointers to activation functions
	std::vector<func_ptr_acivation> D_Activation_Function_Vect;     //Holds all pointers to the derivatives of the activation functions

	using func_ptr_cost = double(*)(double, double);
	std::vector<func_ptr_cost> Cost_Function_Vect;				//Holds all pointers to cost functions
	std::vector<func_ptr_cost> D_Cost_Function_Vect;			//Holds all pointers to the derivatives of the activation functions


	//Holds all the warning messages 
	std::vector<std::string> Warning_Messages;

public:
	//Network vector
	/*
	* Important Things to Know
	*	- Layer refers to each individual slice of the network
	*	- Position refers to a node inside of a layer
	*/
	std::vector<layer> Network;

	/*
	* Startup Code
	*/
	//Initiallizing functions
	Neural_Network();

	/*
	* Functions to Create the Network
	*/

	//Settings
	double Setting_No_Rand_Bias = 0;	//Holds a standard biase value used to assign all biases when intiallized 
	double Setting_No_Rand_Weights = 0; //Holds a standard weight value used to assign all weights when intiallized

	//Adds a layer to the end of the network
	void Add_Layer(unsigned number_of_nodes, std::vector<unsigned>* activation_functions);

	//Adds a layer anywhere in the network
	void Add_Layer(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions);

	//Adding individual connections in the network
	void Add_Node(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions);

	//Adds an extra_connection (struct) to two nodes in the same network
	void Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position);

	//Adds an extra_connection (struct) to two nodes in different networks
	//The first two inputs indicate where in this network the node is located, the third input is a pointer to the node in the other network
	void Monster_Add_Connection(unsigned first_node_layer, unsigned first_node_position, node* node_in_other_network);

	//Assigns cost functions to the final layer with one vector
	void Add_Cost_Function(std::vector<unsigned>* cost_functions);

	//Assigns specific nodes cost functions
	void Add_Cost_Function(unsigned layer_in_network, unsigned node_in_layer, unsigned cost_function_number);

	//Deletes nodes ability to have a cost function
	void Remove_Cost_Function(unsigned layer_in_network, unsigned node_in_layer);

	//Connects every node togeher with every node in other layers
	void Monster_Connect_All();

	/*
	* Functions to initialize and edit network settings
	*/

	double Rand_High_Range = 1; double Rand_Low_Range = -1;  double Rand_Decimal_Multiplier = 1000;
	double random_generator();

	//Randomizes all weight values
	void Randomize_weight_values();

	//Randomizes biases of all nodes except in first layer
	void Randomize_bias_values();

	//Randomizes biases of all nodes including first layer
	void Monster_Randomize_bias_values();

	/*
	* Functions to create "husk networks" (networks without any forward or back propogation)
	* The purpose of husk networks is to store previous network information
	*/

	//Creates weighted connections between husk network and present network
	void Monster_Husk_Create_Network_Connection(Neural_Network* husk_network);

	//Stores node values from present network to a husk network
	void Monster_Husk_Store_Network(Neural_Network* husk_network);

	/*
	* Cost Evaluating Functions
	*/

	double Network_Cost(std::vector<double>* output);

	/*
	* Propogating Functions to Use the Network
	*/

	//Forward propogates the network normally
	void Forward_Propogate(std::vector<double>* input);

	//Forward propogates the network with inputs from other networks nodes
	void Monster_Forward_Propogate();

	//Back propogates the network using gradient descent algorithm
	//output vector holds all correct output starting in the top output node of the network and is ordered from there to the bottom input node of the network
	double Learning_Rate = 0.001;
	void Back_Propogate_Gradient(std::vector<double>* output);

	//Back propogates the network based on the back propogation of a another network
	void Monster_Back_Propograte_Gradient();


	/*
	* Error Handeling Functions
	*	- Location format "file\function\#identifier in function"
	*/

	//Holds all the error messages so that if there is an error it can be called and will exit out of any program
	//All Error messages will go on cerr
	void Error(unsigned code, std::string location);

	//All Warnings messages will go on cout and are from Warning_Messages vector
	void Warning(unsigned code, std::string location);


	/*
	* Print Functions
	*/
	//Prints out a standard space to keep spacing the same
	const std::string space_interval = "     ";
	void Space();
	void Space(unsigned number_of_spaces);


	//Prints the whole make up of the network
	void Print_Whole_Network();

	//Prints the output of the final code
	void Print_Node_Output();

	//Prints the cost at each node
	void Print_Cost_Nodes(std::vector<double>* output);

	void Print_All_D_Acts(double value);
};

#endif