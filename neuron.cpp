#include "neuron.hpp"
using namespace std;

Neuron::Neuron(double b) {
	bias = b;
}

void Neuron::setBias(double b) {
	bias = b;
}

double Neuron::getBias() {
	return bias;
}