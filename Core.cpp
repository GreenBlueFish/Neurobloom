#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <sstream>

#include "Core.h"

//SupportFunctions
#include "Activation_Functions.h" //for forward and back propogation
#include "Cost_Functions.h"       //for evaluating network outputs

//Macro files
#include "C:\Users\rowan\Documents\Coding_Projects\Joels_Code\MacroTools.h"

//Constructor for initial start up sequence
Neural_Network::Neural_Network() {
    //Adding activation functions and their derivatives to their vectors
    {
        Activation_Function_Vect.push_back(Activation::Null);            //0         
        Activation_Function_Vect.push_back(Activation::Sigmoid);         //1
        Activation_Function_Vect.push_back(Activation::TanH);            //2
        Activation_Function_Vect.push_back(Activation::ReLU);            //3
        Activation_Function_Vect.push_back(Activation::Leaky_ReLU);      //4
        Activation_Function_Vect.push_back(Activation::Sine);            //5
        Activation_Function_Vect.push_back(Activation::Cosine);          //6

        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_Null);
        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_Sigmoid);
        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_TanH);
        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_ReLU);
        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_Leaky_ReLU);
        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_Sine);
        D_Activation_Function_Vect.push_back(Activation::DERIVATIVE_Cosine);

        //Error handeling
        if (Activation_Function_Vect.size() != D_Activation_Function_Vect.size()) Error(4, "Neural_Network::Neural_Network()");
    }

    //Adding cost functions
    {
        Cost_Function_Vect.push_back(Cost::Difference_Squared);    //0

        D_Cost_Function_Vect.push_back(Cost::DERIVATIVE_Difference_Squared);

        //Error handeling
        if (Cost_Function_Vect.size() != D_Cost_Function_Vect.size()) Error(4, "Neural_Network::Neural_Network()");
    }

    //Loading Warning Messages into program (this is not done for Error_Messages because they can only be called once, then the code is terminated
    {
        std::ifstream Warning_Messages_File("Codes\\Warning_Messages.txt");

        //Making sure the file opens properly
        if (!Warning_Messages_File.is_open()) {
            //Red Color 
            setTextColorRGB(255, 0, 0);
            std::cerr << "ERROR opening \"Warning_Messages.txt\" File" << std::endl;

            //Resetting Output Colour
            setTextColorRGB(255, 255, 255);

            exit(0);
        }

        //Loading Warning Messages
        std::string warning_message_line = "";
        while (getline(Warning_Messages_File, warning_message_line)) {
            Warning_Messages.push_back(warning_message_line);
        }
        Warning_Messages_File.close();
    }
}

/*
* Functions to Create the Network
*/

//Adds a layer to the end of the network
void Neural_Network::Add_Layer(unsigned number_of_nodes, std::vector<unsigned>* activation_functions) {
    //Error handling
    if (activation_functions->size() != number_of_nodes) Error(0, "void Neural_Network::Add_Layer(unsigned number_of_nodes, std::vector<unsigned>* activation_functions)");

    //Warning handling

    //Function Code

    //Temp variables
    layer temp_layer;
    node temp_node;

    //Adding nodes
    for (unsigned i = 0; i < number_of_nodes; i++) {
        //Changing activation function
        temp_node.Act_func = (*activation_functions)[i];

        //Initiallizing bias
        temp_node.bias = Setting_No_Rand_Bias;

        //Adding node
        temp_layer.layer_nodes.push_back(temp_node);
    }

    //If the network is being initiallized (no layers are present) set all biases to zero
    if (Network.size() == 0) {
        for (int n = 0; n < temp_layer.layer_nodes.size(); n++) temp_layer.layer_nodes[n].bias = 0;
    }
    //If the network has already been initiallied a 2d weight matrix must be made
    else {
        //Initiallizing weights with Setting_No_Rand_Weights
        std::vector<std::vector<double>> temp_weights(number_of_nodes, std::vector<double>(Network[Network.size() - 1].layer_nodes.size(), Setting_No_Rand_Weights));

        //Adding weights
        temp_layer.weights = temp_weights;

        //Deleting weight vector for memory
        temp_weights.clear();

        //setting has_cost to false in previous layer
        for (int n = 0; n < Network[Network.size() - 1].layer_nodes.size(); n++) {
            Network[Network.size() - 1].layer_nodes[n].has_cost = false;
        }
    }
    Network.push_back(temp_layer);
}

//Adds a layer anywhere in the network
void Neural_Network::Add_Layer(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions) {
    //Error handling
    if (activation_functions->size() != number_of_nodes) Error(0, "void Neural_Network::Add_Layer(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions)");

    //Warning handling
    if (layer_in_network == 0) Warning(0, "Neural_Network::Add_Layer(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions)");

    //Function Code

    //Temp variables
    layer temp_layer;
    node temp_node;

    //Adding nodes
    for (unsigned i = 0; i < number_of_nodes; i++) {
        //Changing activation function
        temp_node.Act_func = (*activation_functions)[i];

        //Initiallizing bias
        temp_node.bias = Setting_No_Rand_Bias;

        //Assumming the node isn't in the last layer (corrrected if the last layer condition is executed
        temp_node.has_cost = false;

        //Adding node
        temp_layer.layer_nodes.push_back(temp_node);
    }


    //If insertion is at the very front of the network
    if (layer_in_network == 0) {
        //resetting all biases to zero and setting all has_cost's to false
        for (int n = 0; n < temp_layer.layer_nodes.size(); n++) temp_layer.layer_nodes[n].bias = 0;

        //Initiallizing weights for front layer to work with new inserted front layer
        std::vector<std::vector<double>> temp_weights(Network[0].layer_nodes.size(), std::vector<double>(number_of_nodes, Setting_No_Rand_Weights));
        Network[0].weights = temp_weights;

        //Inserting new layer
        Network.insert(Network.begin(), temp_layer);
    }
    //If insertion is in the middle of the network
    else if ((0 < layer_in_network) && (layer_in_network < (Network.size() - 1))) {
        //Initiallizing weights for existing layer to work with new inserted front layer
        std::vector<std::vector<double>> temp_weights_existing(Network[layer_in_network].layer_nodes.size(), std::vector<double>(number_of_nodes, Setting_No_Rand_Weights));
        Network[layer_in_network].weights = temp_weights_existing;
        temp_weights_existing.clear();

        //Initiallizing weights for new layer to work with existing forward layer
        std::vector<std::vector<double>> temp_weights_new(number_of_nodes, std::vector<double>(Network[layer_in_network - 1].layer_nodes.size(), Setting_No_Rand_Weights));
        temp_layer.weights = temp_weights_new;
        temp_weights_new.clear();

        //Inserting new layer
        Network.insert(Network.begin() + layer_in_network, temp_layer);
    }
    //If insertion is at the end of the network
    else {
        //Initiallizing weights with Setting_No_Rand_Weights to end of network
        std::vector<std::vector<double>> temp_weights(number_of_nodes, std::vector<double>(Network[Network.size() - 1].layer_nodes.size(), Setting_No_Rand_Weights));

        //Adding weights
        temp_layer.weights = temp_weights;

        //Deleting weight vector for memory
        temp_weights.clear();

        //Setting all final nodes to have cost function
        for (int n = 0; n < temp_layer.layer_nodes.size(); n++) {
            temp_layer.layer_nodes[n].has_cost = true;
        }

        //Inserting new layer at end of network
        Network.push_back(temp_layer);
    }
}

void Neural_Network::Add_Node(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions) {
    //Error handling
    if (activation_functions->size() != number_of_nodes) Error(0, "void Neural_Network::Add_Node(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions)");
    if (layer_in_network >= Network.size()) Error(0, "void Neural_Network::Add_Node(unsigned layer_in_network, unsigned number_of_nodes, std::vector<unsigned>* activation_functions)");

    //Warning handling

    //Function Code

    node temp_node;

    for (int n = 0; n < number_of_nodes; n++) {
        //Changing activation function
        temp_node.Act_func = (*activation_functions)[n];
        temp_node.has_cost = false;

        //Adding node
        Network[layer_in_network].layer_nodes.push_back(temp_node);
    }

    //Adding to the first layer (the Network.size() is to prevent a condition from happening to the else in these else if statements)
    if ((layer_in_network == 0) || (Network.size() == 0)) {
        if (Network.size() != 0) {
            std::vector<std::vector<double>> temp_weights_existing(Network[0].layer_nodes.size(), std::vector<double>(Network[1].layer_nodes.size(), Setting_No_Rand_Weights));
            Network[0].weights = temp_weights_existing;
            temp_weights_existing.clear();
        }
    }
    //If insertion is in the middle of the network
    else if ((0 < layer_in_network) && (layer_in_network < (Network.size() - 1))) {
        std::vector<std::vector<double>> temp_weights_existing(Network[layer_in_network + 1].layer_nodes.size(), std::vector<double>(Network[layer_in_network].layer_nodes.size(), Setting_No_Rand_Weights));
        Network[layer_in_network + 1].weights = temp_weights_existing;
        temp_weights_existing.clear();

        std::vector<std::vector<double>> temp_weights_new(Network[layer_in_network].layer_nodes.size(), std::vector<double>(Network[layer_in_network - 1].layer_nodes.size(), Setting_No_Rand_Weights));
        Network[layer_in_network].weights = temp_weights_new;
        temp_weights_new.clear();
    }
    //If insertion is at the end of the network
    else {
        //Setting all final nodes to have cost function
        for (int n = 0; n < Network[Network.size() - 1].layer_nodes.size(); n++) {
            Network[Network.size() - 1].layer_nodes[n].has_cost = true;
        }

        std::vector<std::vector<double>> temp_weights_new(Network[Network.size() - 1].layer_nodes.size(), std::vector<double>(Network[Network.size() - 2].layer_nodes.size(), Setting_No_Rand_Weights));
        Network[Network.size() - 1].weights = temp_weights_new;
        temp_weights_new.clear();
    }
}

//Adds an extra_connection (struct) to two nodes in the network
void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position) {
    //Error handling
    if (first_node_layer >= Network.size()) Error(0, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position)");
    if (second_node_layer >= Network.size()) Error(0, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position)");

    if (first_node_position >= Network[first_node_layer].layer_nodes.size()) Error(0, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position)");
    if (second_node_position >= Network[second_node_position].layer_nodes.size()) Error(0, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position)");

    if (first_node_layer == second_node_layer) Error(1, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, unsigned second_node_layer, unsigned second_node_position)");

    //Warning handling

    //Function Code

    extra_connection temp;

    //Creating connections
    if (first_node_layer < second_node_layer) {
        temp.weight = Setting_No_Rand_Weights;
        temp.node_connection = &Network[first_node_layer].layer_nodes[first_node_position];
        Network[second_node_layer].layer_nodes[second_node_position].backward_extra.push_back(temp);
    }
    else {
        temp.weight = Setting_No_Rand_Weights;
        temp.node_connection = &Network[second_node_layer].layer_nodes[second_node_position];
        Network[first_node_layer].layer_nodes[first_node_position].backward_extra.push_back(temp);
    }
}

//Adds an extra_connection (struct) to two nodes in different networks
//The first two inputs indicate where in this network the node is located, the third input is a pointer to the node in the other network
void Neural_Network::Monster_Add_Connection(unsigned first_node_layer, unsigned first_node_position, node* node_in_other_network) {
    //Error handling
    if (first_node_layer >= Network.size()) Error(0, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, node* node_in_other_network)");
    if (first_node_position >= Network[first_node_layer].layer_nodes.size()) Error(0, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, node* node_in_other_network)");

    if (&Network[first_node_layer].layer_nodes[first_node_position] == node_in_other_network) Error(12, "void Neural_Network::Add_Connection(unsigned first_node_layer, unsigned first_node_position, node* node_in_other_network)");
    
    //Warning handling

    //Function Code

    extra_connection temp;

    temp.weight = Setting_No_Rand_Weights;
    temp.node_connection = node_in_other_network;

    //Making sure that a connection has not already been added
    for (unsigned extra = 0; extra < Network[first_node_layer].layer_nodes[first_node_position].backward_extra.size(); extra++) {
        if (temp.node_connection == Network[first_node_layer].layer_nodes[first_node_position].backward_extra[extra].node_connection) return;
    }

    Network[first_node_layer].layer_nodes[first_node_position].backward_extra.push_back(temp);
}

//Assigns cost functions to the final layer with one vector
void Neural_Network::Add_Cost_Function(std::vector<unsigned>* cost_functions) {
    //Error handling
    if (cost_functions->size() != Network[Network.size() - 1].layer_nodes.size()) Error(6, "void Neural_Network::Add_Cost_Function(std::vector<unsigned>* cost_functions)");

    for (int i = 0; i < cost_functions->size(); i++) {
        if ((*cost_functions)[i] >= Cost_Function_Vect.size()) Error(3, "void Neural_Network::Add_Cost_Function(std::vector<unsigned>* cost_functions)");
    }

    //Warning handling

    //Function Code

    //Assiging cost functions
    for (int n = 0; n < Network[Network.size() - 1].layer_nodes.size(); n++) {
        Network[Network.size() - 1].layer_nodes[n].Cost_func = (*cost_functions)[n];
    }
}

//Assigns specific nodes cost functions
void Neural_Network::Add_Cost_Function(unsigned layer_in_network, unsigned node_in_layer, unsigned cost_function_number) {
    //Error handling
    if (cost_function_number >= Cost_Function_Vect.size()) Error(7, "void Neural_Network::Add_Cost_Function(unsigned layer_in_network, unsigned node_in_layer, unsigned cost_function_number)");
    if (layer_in_network >= Network.size()) Error(8, "void Neural_Network::Add_Cost_Function(unsigned layer_in_network, unsigned node_in_layer, unsigned cost_function_number)");
    if (node_in_layer >= Network[layer_in_network].layer_nodes.size()) Error(9, "void Neural_Network::Add_Cost_Function(unsigned layer_in_network, unsigned node_in_layer, unsigned cost_function_number)");

    //Warning handling

    //Function Code
    Network[layer_in_network].layer_nodes[node_in_layer].has_cost = true;
    Network[layer_in_network].layer_nodes[node_in_layer].Cost_func = cost_function_number;
}

//Deletes nodes ability to have a cost function
void Neural_Network::Remove_Cost_Function(unsigned layer_in_network, unsigned node_in_layer) {
    //Error handling
    if (layer_in_network >= Network.size()) Error(8, "void Neural_Network::Delete_Cost_Function(unsigned layer_in_network, unsigned node_in_layer)");
    if (node_in_layer >= Network[layer_in_network].layer_nodes.size()) Error(9, "void Neural_Network::Delete_Cost_Function(unsigned layer_in_network, unsigned node_in_layer)");

    //Warning handling

    //Function Code
    Network[layer_in_network].layer_nodes[node_in_layer].has_cost = false;
}

//Connects every node togeher with every node in other layers
void Neural_Network::Monster_Connect_All() {
    //Error handling
    if (Network.size() <= 0) Error(2, "void Neural_Network::Monster_Connect_All())");

    //Warning handling

    //Function Code

    //Connecting all nodes together
    for (unsigned l_outside = 0; l_outside < Network.size(); l_outside++) {
        for (unsigned l_inside = (l_outside + 1); l_inside < Network.size(); l_inside++) {
            for (unsigned n_outside = 0; n_outside < Network[l_outside].layer_nodes.size(); n_outside++) {
                for (unsigned n_inside = 0; n_inside < Network[l_inside].layer_nodes.size(); n_inside++) {
                    Add_Connection(l_outside, n_outside, l_inside, n_inside);
                }
            }
        }
    }
}

/*
* Functions to initialize and edit network settings
*/

double Neural_Network::random_generator()
{
    int high_range = Rand_High_Range * Rand_Decimal_Multiplier;
    int low_range = Rand_Low_Range * Rand_Decimal_Multiplier;
    double random = (rand() % (high_range - low_range)) / Rand_Decimal_Multiplier;

    return (random + Rand_Low_Range);
}

//Randomizes all weight values
void Neural_Network::Randomize_weight_values() {
    for (int l = 0; l < Network.size(); l++) {
        for (int n_current = 0; n_current < Network[l].layer_nodes.size(); n_current++) {
            //Randomizing weights to previous layer
            if (l != 0) {
                for (int n_previous = 0; n_previous < Network[l].weights[n_current].size(); n_previous++) {
                    Network[l].weights[n_current][n_previous] = random_generator();
                }
            }
            double random_weight;

            //Randomizing weights to extra connections
            for (int w = 0; w < Network[l].layer_nodes[n_current].backward_extra.size(); w++) {
                Network[l].layer_nodes[n_current].backward_extra[w].weight = random_generator();
            }
        }
    }
}

//Randomizes biases of all nodes except in first layer
void Neural_Network::Randomize_bias_values() {
    for (int l = 1; l < Network.size(); l++) {
        for (int n = 0; n < Network[l].layer_nodes.size(); n++) {
            Network[l].layer_nodes[n].bias = random_generator();
        }
    }
}

//Randomizes biases of all nodes including first layer
void Neural_Network::Monster_Randomize_bias_values() {
    for (int l = 0; l < Network.size(); l++) {
        for (int n = 0; n < Network[l].layer_nodes.size(); n++) {
            Network[l].layer_nodes[n].bias = random_generator();
        }
    }
}

/*
* Functions to create and use "husk networks" (networks without any forward or back propogation)
* The purpose of husk networks is to store previous network information
*/

//Creates weighted connections between husk network and present network
void Neural_Network::Monster_Husk_Create_Network_Connection(Neural_Network* husk_network) {
    //Error handling
    if (husk_network->Network.size() != Network.size()) Error(13, "void Monster_Store_Network_to_Husk(Neural_Network* husk_network)");

    //Warning handling
    if (husk_network == nullptr) Warning(1, "void Monster_Store_Network_to_Husk(Neural_Network* husk_network)");

    //Function Code

    //To make this function run quicker the error check for the same number of nodes per layer and the updating of the nodes 
    //is done in the same loop
    for (unsigned l = 0; l < Network.size(); l++) {
        if (husk_network->Network[l].layer_nodes.size() != Network[l].layer_nodes.size()) Error(13, "void Monster_Store_Network_to_Husk(Neural_Network* husk_network)");
        for (unsigned n = 0; n < Network[l].layer_nodes.size(); n++) {
            Monster_Add_Connection(l,n, &husk_network->Network[l].layer_nodes[n]);
        }
    }
}

//Stores node values from present network to a husk network
//Node values are stored in value)act in the husk network
void Neural_Network::Monster_Husk_Store_Network(Neural_Network* husk_network) {
    //Error handling
    if (husk_network->Network.size() != Network.size()) Error(13, "void Monster_Store_Network_to_Husk(Neural_Network* husk_network)");

    //Warning handling
    if (husk_network == nullptr) Warning(1, "void Monster_Store_Network_to_Husk(Neural_Network* husk_network)");

    //Function Code

    //To make this function run quicker the error check for the same number of nodes per layer and the updating of the nodes 
    //is done in the same loop
    for (unsigned l = 0; l< Network.size(); l++) {
        if (husk_network->Network[l].layer_nodes.size() != Network[l].layer_nodes.size()) Error(13, "void Monster_Store_Network_to_Husk(Neural_Network* husk_network)");
        for (unsigned n = 0; n < Network[l].layer_nodes.size(); n++) {
            husk_network->Network[l].layer_nodes[n].value_act = Network[l].layer_nodes[n].value_act;
        }
    }
}

/*
* Cost Evaluating Functions
*/

//Returns cost of the network nodes in the present state
double Neural_Network::Network_Cost(std::vector<double>* output) {
    double cost = 0;
    unsigned output_counter = 0;

    //Going through all nodes to see if it has a cost function, evaluate if it does
    for (unsigned l = (Network.size() - 1); l > 0; l--) {
        for (unsigned n = 0; n < Network[l].layer_nodes.size(); n++) {
            if (Network[l].layer_nodes[n].has_cost) {
                if (output_counter >= (*output).size()) Error(11, "double Neural_Network::Network_Cost(std::vector<double>* output)");

                cost += Cost_Function_Vect[Network[l].layer_nodes[n].Cost_func]((*output)[output_counter], Network[l].layer_nodes[n].value_act);
                output_counter++;
            }
        }
    }

    return cost;
}

/*
* Propogating Functions to Use the Network
*/

//Forward propogates the network
void Neural_Network::Forward_Propogate(std::vector<double>* input) {
    //Error handling
    if (input->size() != Network[0].layer_nodes.size()) Error(0, "void Neural_Network::Forward_Propogate(std::vector<double> input)");
    if (Network.size() == 0) Error(2, "void Neural_Network::Forward_Propogate(std::vector<double> input)");

    //Warning handling

    //Function Code

    //Setting the inputs
    for (int n = 0; n < Network[0].layer_nodes.size(); n++) {
        Network[0].layer_nodes[n].value_un_act = (*input)[n];

        //Not preforming activation function as this isn't done for the first layer
        Network[0].layer_nodes[n].value_act = (*input)[n];
    }

    //Forward propagating
    for (int l = 1; l < Network.size(); l++) {
        for (int n_current = 0; n_current < Network[l].layer_nodes.size(); n_current++) {
            //Resetting node value
            Network[l].layer_nodes[n_current].value_un_act = 0;

            //Adding all node * weight values from previous layer to current node
            for (int n_previous = 0; n_previous < Network[l].weights[n_current].size(); n_previous++) {
                Network[l].layer_nodes[n_current].value_un_act += (Network[l - 1].layer_nodes[n_previous].value_act * Network[l].weights[n_current][n_previous]);
            }

            //Adding all node * weight values from extra connections
            for (int extra = 0; extra < Network[l].layer_nodes[n_current].backward_extra.size(); extra++) {
                Network[l].layer_nodes[n_current].value_un_act += (Network[l].layer_nodes[n_current].backward_extra[extra].node_connection->value_act * Network[l].layer_nodes[n_current].backward_extra[extra].weight);
            }

            //Adding bias
            Network[l].layer_nodes[n_current].value_un_act += Network[l].layer_nodes[n_current].bias;

            //Preforming activation function
            Network[l].layer_nodes[n_current].value_act = Activation_Function_Vect[Network[l].layer_nodes[n_current].Act_func](Network[l].layer_nodes[n_current].value_un_act);
        }
    }
}

//Forward propogates the network with inputs from other networks nodes
void Neural_Network::Monster_Forward_Propogate() {
    //Error handling
    if (Network.size() <= 0) Error(2, "void Neural_Network::Forward_Propogate_Monster()");

    //Warning handling

    //Function Code

    //Forward propogating first layer which is made up of only connections to another network
    for (unsigned n = 0; n < Network[0].layer_nodes.size(); n++) {
        //Resetting node value
        Network[0].layer_nodes[n].value_un_act = 0;

        //Adding all connection node values * weight
        for (unsigned n_connected = 0; n_connected < Network[0].layer_nodes[n].backward_extra.size(); n_connected++) {
            Network[0].layer_nodes[n].value_un_act += (Network[0].layer_nodes[n].backward_extra[n_connected].weight * Network[0].layer_nodes[n].backward_extra[n_connected].node_connection->value_act);
        }

        //Adding bias
        Network[0].layer_nodes[n].value_un_act += Network[0].layer_nodes[n].bias;

        //Applying activation function
        Network[0].layer_nodes[n].value_act = Activation_Function_Vect[Network[0].layer_nodes[n].Act_func](Network[0].layer_nodes[n].value_un_act);
    }

    //Forward propogating all layers except first
    for (unsigned l = 1; l < Network.size(); l++) {
        for (unsigned n_current = 0; n_current < Network[l].layer_nodes.size(); n_current++) {
            //Resetting node value
            Network[l].layer_nodes[n_current].value_un_act = 0;

            //Adding all node * weight values from previous layer to current node
            for (int n_previous = 0; n_previous < Network[l].weights[n_current].size(); n_previous++) {
                Network[l].layer_nodes[n_current].value_un_act += (Network[l - 1].layer_nodes[n_previous].value_act * Network[l].weights[n_current][n_previous]);
            }

            //Adding all node * weight values from extra connections
            for (int extra = 0; extra < Network[l].layer_nodes[n_current].backward_extra.size(); extra++) {
                Network[l].layer_nodes[n_current].value_un_act += (Network[l].layer_nodes[n_current].backward_extra[extra].node_connection->value_act * Network[l].layer_nodes[n_current].backward_extra[extra].weight);
            }

            //Adding bias
            Network[l].layer_nodes[n_current].value_un_act += Network[l].layer_nodes[n_current].bias;

            //Preforming activation function
            Network[l].layer_nodes[n_current].value_act = Activation_Function_Vect[Network[l].layer_nodes[n_current].Act_func](Network[l].layer_nodes[n_current].value_un_act);

        }
    }
}

//Back propogates the network using gradient descent algorithm
//output vector holds all correct output starting in the top output node of the network and is ordered from there to the bottom input node of the network
void Neural_Network::Back_Propogate_Gradient(std::vector<double>* output) {
    //I use this number a couple times so I only want to calculate it one
    unsigned Net_size = Network.size() - 1;

    //Error handling
    if (output->size() < Network[Net_size].layer_nodes.size()) Error(10, "void Neural_Network::Back_Propogate_Gradient(std::vector<double>* output)");
    if (Network.size() == 0) Error(2, "void Neural_Network::Back_Propogate_Gradient(std::vector<double>* output)");

    for (unsigned n = 0; n < Network[Net_size].layer_nodes.size(); n++) {
        if (Network[Net_size].layer_nodes[n].Cost_func >= Cost_Function_Vect.size()) Error(3, "void Neural_Network::Back_Propogate_Gradient(std::vector<double>* output)");
    }

    //Warning handling

    //Function Code

    //Gradient descent back propogation
    unsigned output_counter = 0;
    double temp_data_holder = 0;              //Holds data
    double temp_data_holder_multiply = 1;

    //Calculating cost for all nodes of last layer
    for (unsigned n = 0; n < Network[Net_size].layer_nodes.size(); n++) {
        Network[Net_size].layer_nodes[n].back_prop_data = D_Cost_Function_Vect[Network[Net_size].layer_nodes[n].Cost_func]((*output)[output_counter], Network[Net_size].layer_nodes[n].value_act);
        std::cout << "Node: " << n << std::endl;
        Space(); std::cout << "D_Cost_Function: " << D_Cost_Function_Vect[Network[Net_size].layer_nodes[n].Cost_func]((*output)[output_counter], Network[Net_size].layer_nodes[n].value_act) << "    Output: " << (*output)[output_counter] << "    Value_Act: " << Network[Net_size].layer_nodes[n].value_act << "    back_prop_data: " << Network[Net_size].layer_nodes[n].back_prop_data << std::endl;
        output_counter++;
    }

    std::cout << std::endl << "Full Back Prop" << std::endl;

    for (unsigned l = Net_size; l > 0; l--) {
        for (unsigned n_current = 0; n_current < Network[l].layer_nodes.size(); n_current++) {

            Space(); std::cout << "Layer: " << l << std::endl;
            Space(2); std::cout << "Node Current: " << n_current << std::endl;

            /*
            * Back Prop Data Setup
            */

            //setting back_prop_data equal to the derivative of the act function at that node
            Network[l].layer_nodes[n_current].back_prop_data = D_Activation_Function_Vect[Network[l].layer_nodes[n_current].Act_func](Network[l].layer_nodes[n_current].value_un_act) * temp_data_holder_multiply;

            Space(3); std::cout << "D_Activation_Function: " << D_Activation_Function_Vect[Network[l].layer_nodes[n_current].Act_func](Network[l].layer_nodes[n_current].value_un_act) << "    Act_func: " << Network[l].layer_nodes[n_current].Act_func << "    Value_Un_Act: " << Network[l].layer_nodes[n_current].value_un_act << "    back_prop_data: " << Network[l].layer_nodes[n_current].back_prop_data << std::endl;


            //multiplying by the cost function if there is one to the back_prop_data
            output_counter = 0;
            if (Network[l].layer_nodes[n_current].has_cost) {
                if (output_counter >= (*output).size()) Error(11, "void Neural_Network::Back_Propogate_Gradient(std::vector<double>* output)");
                
                Space(3); std::cout << "Back_Prop_Data before: " << Network[l].layer_nodes[n_current].back_prop_data;
                Network[l].layer_nodes[n_current].back_prop_data *= D_Cost_Function_Vect[Network[l].layer_nodes[n_current].Cost_func]((*output)[output_counter], Network[l].layer_nodes[n_current].value_act);
               
                std::cout << "    Back_Prop_Data after: " << Network[l].layer_nodes[n_current].back_prop_data << "    D_Cost_Function_Vect: " << D_Cost_Function_Vect[Network[l].layer_nodes[n_current].Cost_func]((*output)[output_counter], Network[l].layer_nodes[n_current].value_act) << "    Output: " << (*output)[output_counter]  << "    Value_Act: " << Network[l].layer_nodes[n_current].value_act << std::endl;
                
                output_counter++;
            }

            //For next layer's back_prop_data
            temp_data_holder += Network[l].layer_nodes[n_current].back_prop_data;

            Space(3); std::cout << "Temp_data_holder: " << temp_data_holder << "    back_prop_data: " << Network[l].layer_nodes[n_current].back_prop_data << std::endl;

            /*
            * Calculating new bias and weight values
            */

            //Updating bias values
            Space(3); std::cout << "Bias before: " << Network[l].layer_nodes[n_current].bias;

            Network[l].layer_nodes[n_current].bias -= (Learning_Rate * Network[l].layer_nodes[n_current].back_prop_data);

            std::cout << "    Bias after: " << Network[l].layer_nodes[n_current].bias << "    learning_rate: " << Learning_Rate << "    back_prop_data: " << Network[l].layer_nodes[n_current].back_prop_data << std::endl;

            //Updating weight values
            for (unsigned n_previous = 0; n_previous < Network[l-1].layer_nodes.size(); n_previous++) {
                Space(3); std::cout << "Learning infront: " << (Learning_Rate * Network[l].layer_nodes[n_current].back_prop_data * Network[l - 1].layer_nodes[n_previous].value_act) << std::endl;
                Space(3); std::cout << "Learning behind:  " << (Network[l].layer_nodes[n_current].back_prop_data * Network[l - 1].layer_nodes[n_previous].value_act * Learning_Rate) << std::endl;

                Space(3); std::cout << "Weight before: " << Network[l].weights[n_current][n_previous];
                Network[l].weights[n_current][n_previous] -= (Learning_Rate * Network[l].layer_nodes[n_current].back_prop_data * Network[l - 1].layer_nodes[n_previous].value_act);

                std::cout << "    Weight after: " << Network[l].weights[n_current][n_previous] << "    back_prop_data: " << Network[l].layer_nodes[n_current].back_prop_data << "    n_previous: " << n_previous << "    n_previous value_act: " << Network[l - 1].layer_nodes[n_previous].value_act << std::endl;
            }

            for (unsigned extra = 0; extra < Network[l].layer_nodes[n_current].backward_extra.size(); extra++) {
                Space(3); std::cout << "Extra weight before: " << Network[l].layer_nodes[n_current].backward_extra[extra].weight;
                Network[l].layer_nodes[n_current].backward_extra[extra].weight -= (Learning_Rate * Network[l].layer_nodes[n_current].back_prop_data * Network[l].layer_nodes[n_current].backward_extra[extra].node_connection->value_act);
                std::cout << Network[l].layer_nodes[n_current].backward_extra[extra].weight << "    back_prop_data: " << Network[l].layer_nodes[n_current].back_prop_data << "extra: " << extra << "    back node value_act: " << Network[l].layer_nodes[n_current].backward_extra[extra].node_connection->value_act << std::endl;
            }
        }
        //updating data holders
        temp_data_holder_multiply = temp_data_holder;
        Space(3); std::cout << "temp_data_holder_multiply: " << temp_data_holder_multiply << std::endl;
        temp_data_holder = 0;
    }

    //updatating input nodes back_prop_data if they are connected to by another network
    for (unsigned n = 0; n < Network[0].layer_nodes.size(); n++) {
        Network[0].layer_nodes[n].back_prop_data = D_Activation_Function_Vect[Network[0].layer_nodes[n].Act_func](Network[0].layer_nodes[n].value_un_act) * temp_data_holder_multiply;

        //multiplying by the cost function if there is one to the back_prop_data
        if (Network[0].layer_nodes[n].has_cost) Network[0].layer_nodes[n].back_prop_data *= D_Cost_Function_Vect[Network[0].layer_nodes[n].Cost_func]((*output)[output_counter], Network[0].layer_nodes[n].value_un_act);

    }
}

/*
* CREATE BELOW FUNCTION
*/

//Back propogates the network based on the back propogation of a another network
void Monster_Back_Propograte_Gradient() {

}

/*
* Error Handeling Functions
*	- Location format "file\function\#identifier in function"
*/

//Holds all the error messages so that if there is an error it can be called and will exit out of any program
//All Error messages will go on cerr
void Neural_Network::Error(unsigned code, std::string location) {
    std::ifstream Error_Messages_File("Codes\\Error_Messages.txt");

    //Making sure the file opens properly
    if (!Error_Messages_File.is_open()) {
        //Red Color 
        setTextColorRGB(255, 0, 0);
        std::cerr << "ERROR opening \"Codes\\Error_Messages.txt\" File" << std::endl;

        //Resetting Output Colour
        setTextColorRGB(255, 255, 255);

        exit(0);
    }

    std::string error_message_line = "";
    unsigned line_count = 0;

    while (getline(Error_Messages_File, error_message_line)) {
        std::cout << "Line: " << line_count << std::endl;
        if (line_count == code) {
            //Red Color 
            setTextColorRGB(255, 0, 0);
            std::cerr << std::endl << "ERROR: " << error_message_line << " at " << location;
            Error_Messages_File.close();

            //Resetting Output Colour
            setTextColorRGB(255, 255, 255);

            exit(0);
        }
        line_count++;
    }
    //Red Color 
    setTextColorRGB(255, 0, 0);
    std::cerr << std::endl << "ERROR: ERROR Code Not Found for ERROR thrown at " << location;
    Error_Messages_File.close();

    //Resetting Output Colour
    setTextColorRGB(255, 255, 255);

    exit(0);
}

//Holds all the warning messages 
//All Warnings messages will go on cout
void Neural_Network::Warning(unsigned code, std::string location) {

    //Checking if the warning message exists
    if (code >= Warning_Messages.size()) {
        //Red Color 
        setTextColorRGB(255, 0, 0);
        std::cerr << std::endl << "ERROR: WARNING Code Not Found for WARNING thrown at " << location;

        //Resetting Output Colour
        setTextColorRGB(255, 255, 255);

        exit(0);
    }

    setTextColorRGB(255, 255, 0);
    std::cout << "WARNING: " << Warning_Messages[code] << " at " << location << std::endl;

    //Resetting Output Colour
    setTextColorRGB(255, 255, 255);
}

/*
* Print Functions
*/

//Prints out a standard space to keep spacing the same
void Neural_Network::Space() {
    std::cout << space_interval;
}
void Neural_Network::Space(unsigned number_of_spaces) {
    for (unsigned i = 0; i < number_of_spaces; i++) {
        std::cout << space_interval;
    }
}

//Prints the whole make up of the network
void Neural_Network::Print_Whole_Network() {
    if (Network.size() == 0) {
        std::cout << "Network has no layers" << std::endl;
    }

    //Going through network
    for (unsigned l = 0; l < Network.size(); l++) {
        std::cout << "Layer: " << l << std::endl;

        for (unsigned n = 0; n < Network[l].layer_nodes.size(); n++) {
            Space(); std::cout << "Node: " << n << "   Address: " << &Network[l].layer_nodes[n] << std::endl;
            Space(2); std::cout << "Act Func # : " << Network[l].layer_nodes[n].Act_func << std::endl;
            Space(2); std::cout << "Value Act  : " << Network[l].layer_nodes[n].value_act << std::endl;
            Space(2); std::cout << "Value UnAct: " << Network[l].layer_nodes[n].value_un_act << std::endl;

            if (Network[l].layer_nodes[n].has_cost) {
                Space(2); std::cout << "Cost Function # : " << Network[l].layer_nodes[n].Cost_func << std::endl;
            }

            Space(2); std::cout << "Bias: " << Network[l].layer_nodes[n].bias << std::endl;

            //The first layer doesn't have weights associated with it
            if (l > 0) {
                for (unsigned w = 0; w < Network[l].weights[n].size(); w++) {
                    Space(2); std::cout << "W[" << w << "]: " << Network[l].weights[n][w] << std::endl;
                }
            }

            if (Network[l].layer_nodes[n].backward_extra.size() > 0) {
                Space(2); std::cout << "Extra Connections Backward:" << std::endl;
            }

            //Printing all the extra connections
            for (int w = 0; w < Network[l].layer_nodes[n].backward_extra.size(); w++) {
                Space(3); std::cout << "Node: [" << l << "][" << n << "] to Node Address: " << Network[l].layer_nodes[n].backward_extra[w].node_connection << " with Weight: " << Network[l].layer_nodes[n].backward_extra[w].weight << std::endl;
            }

            std::cout << std::endl;
        }
    }
}

//Prints the output of the final code
void Neural_Network::Print_Node_Output() {
    std::cout << "Node Ouptus, [layer][node]: value" << std::endl;
    
    //Going through all nodes to see if it has a cost function, evaluate if it does
    for (unsigned l = (Network.size() - 1); l > 0; l--) {
        for (unsigned n = 0; n < Network[l].layer_nodes.size(); n++) {
            if (Network[l].layer_nodes[n].has_cost) {
                Space();  std::cout << "[" << l << "][" << n << "]: " << Network[l].layer_nodes[n].value_act << std::endl;
            }
        }
    }
}

//Prints the cost at each node
void Neural_Network::Print_Cost_Nodes(std::vector<double>* output) {
    std::cout << "Node Ouptus, [layer][node]: node value, correct value, cost_func, cost" << std::endl;;
    unsigned output_counter = 0;

    //Going through all nodes to see if it has a cost function, evaluate if it does
    for (unsigned l = (Network.size() - 1); l > 0; l--) {
        for (unsigned n = 0; n < Network[l].layer_nodes.size(); n++) {
            if (Network[l].layer_nodes[n].has_cost) {
                if (output_counter >= (*output).size()) Error(11, "void Neural_Network::Print_Cost_Nodes(std::vector<double>* output)");

                Space();  std::cout << "[" << l << "][" << n << "]: " << Network[l].layer_nodes[n].value_act << ", " << (*output)[output_counter] << ", " << Network[l].layer_nodes[n].Cost_func << ", " << Cost_Function_Vect[Network[l].layer_nodes[n].Cost_func]((*output)[output_counter], Network[l].layer_nodes[n].value_act) << std::endl;
                output_counter++;
            }
        }
    }
}

void Neural_Network::Print_All_D_Acts(double value) {
    for (int i = 0; i < 7; i++) {
        std::cout << "Vect Act Func Derivative at " << i << ": " << D_Activation_Function_Vect[i](value) << "                    Act Func: " << Activation_Function_Vect[i](value) << std::endl;
    }
}