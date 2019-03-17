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

OutputNode::OutputNode() {
	outNeuron = nullptr;
}

void OutputNode::setOutputNeuron(NeuronPtr n) {
	outNeuron = n;
}

double OutputNode::getOutput() {
	if (outNeuron != nullptr) {
		return outNeuron->getOutput();
	} else
		return 0.0;
}

int OutputNode::getPrediction() {
	double out = getOutput();
	int ansi = out >= 0.5 ? 1 : 0;
	return ansi;
}

int OutputNode::getType() {
	return 3;
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

void Neuron::addOutput(NodePtr output, int inputNum) {
	removeOutput(output);
	nodeForOutput.push_back(make_pair(output, inputNum));
}

void Neuron::removeOutput(NodePtr output) {
	int foundIdx = -1;
	for (int i = 0; i < nodeForOutput.size(); i++) {
		if (output == nodeForOutput[i].first) {
			foundIdx = i;
		}
	}
	if (foundIdx != -1) {
		nodeForOutput.erase(nodeForOutput.begin() + foundIdx);
	}
}

vector<NodePtr> Neuron::getInputNode() {
	return nodeForInput;
}

vector<NodePtr> Neuron::getOutputNode() {
	vector<NodePtr> outputnodes;
	for (int i = 0; i < nodeForOutput.size(); i++) {
		outputnodes.push_back(nodeForOutput[i].first);
	}
	return outputnodes;
}

vector<pair<NodePtr, int>> Neuron::getOutputNodeWithInputIndex() {
	return nodeForOutput;
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

NeuralNetwork::NeuralNetwork(int iCount, int oCount, Activation func) {
	inputCount = iCount;
	outputCount = oCount;
	defaultActivationFunction = func;
	for (int i = 0; i < inputCount; i++) {
		inputs.push_back(new InputNode());
	}
	for (int i = 0; i < outputCount; i++) {
		OutputNodePtr outNode = new OutputNode();
		outputNodes.push_back(outNode);
		NeuronPtr outNeuron = new Neuron(inputCount, defaultActivationFunction);
		outNeuron->addOutput(dynamic_cast<NodePtr>(outNode), 0);
		outNode->setOutputNeuron(outNeuron);
		for (int j = 0; j < inputCount; j++) {
			outNeuron->setInput(dynamic_cast<NodePtr>(inputs[j]), j);
		}
		allNeuron.push_back(outNeuron);
	}
	traverse();
}

NeuralNetwork::~NeuralNetwork() {
	neuronTraverse.clear();
	for (int i = 0; i < allNeuron.size(); i++) {
		delete allNeuron[i];
	}
	for (int i = 0; i < inputs.size(); i++) {
		delete inputs[i];
	}
	for (int i = 0; i < outputNodes.size(); i++) {
		delete outputNodes[i];
	}
	outputNodes.clear();
	allNeuron.clear();
	inputs.clear();
}

void NeuralNetwork::setActivationFunction(Activation func) {
	defaultActivationFunction = func;
	for (int i = 0; i < allNeuron.size(); i++) {
		allNeuron[i]->setActivationFunction(func);
	}
	feedForward();
}

void NeuralNetwork::traverse() {
	neuronTraverse.clear();
	queue<NeuronPtr> bfsQueue;
	for (int i = 0; i < allNeuron.size(); i++) {
		bool inputNeuron = false;
		NeuronPtr nowneuron = allNeuron[i];
		vector<NodePtr> nowChild = nowneuron->getInputNode();
		for (int i = 0; i < nowChild.size(); i++) {
			if (nowChild[i]->getType() == 1) {
				inputNeuron = true;
				break;
			}
		}
		if (inputNeuron) {
			allNeuron[i]->traverseVisited = true;
			bfsQueue.push(allNeuron[i]);
		}
	}
	while (!bfsQueue.empty()) {
		NeuronPtr nodeNow = bfsQueue.front();
		bfsQueue.pop();
		neuronTraverse.push_back(nodeNow);
		vector<NodePtr> nextNode = nodeNow->getOutputNode();
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
	for (int i = 0; i < neuronTraverse.size(); i++) {
		neuronTraverse[i]->recalculate();
	}
}

NeuronPtr NeuralNetwork::addNeuron() {
	NeuronPtr newNeuron = new Neuron(inputCount, defaultActivationFunction);
	for (int i = 0; i < inputCount; i++) {
		newNeuron->setInput(dynamic_cast<NodePtr>(inputs[i]), i);
	}
	allNeuron.push_back(newNeuron);
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

vector<NeuronPtr> NeuralNetwork::getNeurons() {
	return allNeuron;
}

vector<int> NeuralNetwork::getPrediction() {
	vector<int> ans;
	for (int i = 0; i < outputNodes.size(); i++) {
		int ansi = outputNodes[i]->getPrediction();
		ans.push_back(ansi);
	}
	return ans;
}

vector<double> NeuralNetwork::getOutput() {
	vector<double> ans;
	for (int i = 0; i < outputNodes.size(); i++) {
		ans.push_back(outputNodes[i]->getOutput());
	}
	return ans;
}