#include "NeuralNetwork.hpp"
#include <cmath>
#include <cstdio>
#include <queue>
using namespace std;

double Node::getOutput() {
	return 0.0;
}

int Node::getType() {
	return 0;
}

InputNode::InputNode() {
	value = 0.0;
}

void InputNode::setValue(double n) {
	value = n;
}

double InputNode::getOutput() {
	return value;
}

int InputNode::getType() {
	return 1;
}

Neuron::Neuron(int iSize) {
	traverseVisited = false;
	setInputSize(iSize);
	activator = [](double x) {
		return 1.0 / (1 - exp(x));
	};
}

Neuron::Neuron(int iSize, Activation func) {
	traverseVisited = false;
	setInputSize(iSize);
	activator = func;
}

void Neuron::setActivationFunction(Activation func) {
	activator = func;
}

double Neuron::getNet() {
	return lastNet;
}

double Neuron::getOutput() {
	return lastOutput;
}

int Neuron::getType() {
	return 2;
}

void Neuron::setInput(NodePtr input, int idx) {
	if (idx >= nodeForInput.size())
		return;
	nodeForInput[idx] = input;
	recalculate();
}

vector<NodePtr> Neuron::getInputNode() {
	return nodeForInput;
}

void Neuron::recalculate() {
	lastNet = 0.0;
	//printf("input: ");
	for (int i = 0; i < nodeForInput.size(); i++) {
		if (nodeForInput[i] != nullptr) {
			double inputi = nodeForInput[i]->getOutput();
			//printf("%lf ", inputi);
			lastNet += inputi * theta[i];
		}
	}
	//printf("net: %lf ", lastNet);
	lastOutput = activator(lastNet);
	//printf("out: %lf\n", lastOutput);
}

void Neuron::setInputSize(int size) {
	inputCount = size;
	theta.resize(inputCount, 0.5);
	nodeForInput.resize(inputCount, nullptr);
}

NeuralNetwork::NeuralNetwork(int iCount, int oCount) {
	inputCount = iCount;
	outputCount = oCount;
	for (int i = 0; i < inputCount; i++) {
		inputs.push_back(new InputNode());
	}
	for (int i = 0; i < outputCount; i++) {
		NeuronPtr outNeuron = new Neuron(inputCount);
		for (int j = 0; j < inputCount; j++) {
			outNeuron->setInput(dynamic_cast<NodePtr>(inputs[j]), j);
		}
		outputNeurons.push_back(outNeuron);
	}
	traverse();
}

NeuralNetwork::NeuralNetwork(int iCount, int oCount, Activation func) {
	inputCount = iCount;
	outputCount = oCount;
	defaultActivationFunction = func;
	for (int i = 0; i < inputCount; i++) {
		inputs.push_back(new InputNode());
	}
	for (int i = 0; i < outputCount; i++) {
		NeuronPtr outNeuron = new Neuron(inputCount, defaultActivationFunction);
		for (int j = 0; j < inputCount; j++) {
			outNeuron->setInput(dynamic_cast<NodePtr>(inputs[j]), j);
		}
		outputNeurons.push_back(outNeuron);
	}
	traverse();
}

NeuralNetwork::~NeuralNetwork() {
	outputNeurons.clear();
	hiddenNeurons.clear();
	for (int i = 0; i < neuronTraverse.size(); i++) {
		delete neuronTraverse[i];
	}
	for (int i = 0; i < inputs.size(); i++) {
		delete inputs[i];
	}
	neuronTraverse.clear();
	inputs.clear();
}

void NeuralNetwork::setActivationFunction(Activation func) {
	defaultActivationFunction = func;
	for (int i = 0; i < outputNeurons.size(); i++) {
		outputNeurons[i]->setActivationFunction(func);
	}
	for (int i = 0; i < hiddenNeurons.size(); i++) {
		hiddenNeurons[i]->setActivationFunction(func);
	}
}

void NeuralNetwork::traverse() {
	neuronTraverse.clear();
	queue<NeuronPtr> bfsQueue;
	for (int i = 0; i < outputNeurons.size(); i++) {
		outputNeurons[i]->traverseVisited = true;
		bfsQueue.push(outputNeurons[i]);
	}
	while (!bfsQueue.empty()) {
		NeuronPtr nodeNow = bfsQueue.front();
		bfsQueue.pop();
		neuronTraverse.push_back(nodeNow);
		vector<NodePtr> nextNode = nodeNow->getInputNode();
		for (int i = 0; i < nextNode.size(); i++) {
			if (nextNode[i]->getType() == 2) {
				NeuronPtr nextNeuronNode = dynamic_cast<NeuronPtr>(nextNode[i]);
				if (!nextNeuronNode->traverseVisited) {
					nextNeuronNode->traverseVisited = true;
					bfsQueue.push(nextNeuronNode);
				}
			}
		}
	}
	for (int i = 0; i < neuronTraverse.size(); i++) {
		neuronTraverse[i]->traverseVisited = false;
	}
}

void NeuralNetwork::feedForward() {
	for (int i = neuronTraverse.size() - 1; i >= 0; i--) {
		neuronTraverse[i]->recalculate();
	}
}

NeuronPtr NeuralNetwork::addHiddenNeuron() {
	NeuronPtr newNeuron = new Neuron(inputCount);
	for (int i = 0; i < inputCount; i++) {
		newNeuron->setInput(dynamic_cast<NodePtr>(inputs[i]), i);
	}
	hiddenNeurons.push_back(newNeuron);
	return newNeuron;
}

void NeuralNetwork::setInput(const vector<double>& input) {
	for (int i = 0; i < input.size(); i++) {
		inputs[i]->setValue(input[i]);
	}
	feedForward();
}

void NeuralNetwork::recalculate() {
	traverse();
	feedForward();
}

vector<NeuronPtr> NeuralNetwork::getOutputNeurons() {
	return outputNeurons;
}

vector<NeuronPtr> NeuralNetwork::getHiddenNeurons() {
	return hiddenNeurons;
}

vector<int> NeuralNetwork::getPrediction() {
	vector<int> ans;
	for (int i = 0; i < outputNeurons.size(); i++) {
		int ansi = outputNeurons[i]->getOutput() >= 0.5 ? 1 : 0;
		ans.push_back(ansi);
	}
	return ans;
}

vector<double> NeuralNetwork::getOutput() {
	vector<double> ans;
	for (int i = 0; i < outputNeurons.size(); i++) {
		ans.push_back(outputNeurons[i]->getOutput());
	}
	return ans;
}