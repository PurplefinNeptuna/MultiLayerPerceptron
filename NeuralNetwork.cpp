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

void OutputNode::setTarget(int tar) {
	target = tar;
	recalculate();
}

int OutputNode::getTarget() {
	return target;
}

double OutputNode::getDerrDout() {
	return derrdout;
}

void OutputNode::recalculate() {
	derrdout = getOutput() - (double)target;
}

double OutputNode::getError() {
	return 0.5 * (derrdout) * (derrdout);
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

Neuron::Neuron(int iSize, Activation func, Deactivation dfunc) {
	traverseVisited = false;
	setInputSize(iSize);
	activator = func;
	deactivator = dfunc;
}

void Neuron::setActivationFunction(Activation func, Deactivation dfunc) {
	activator = func;
	deactivator = dfunc;
}

void Neuron::setTheta(int idx, double newtheta) {
	if (idx >= theta.size())
		return;
	theta[idx] = newtheta;
}

void Neuron::setBias(double newbias) {
	bias = newbias;
}

double Neuron::getBias() {
	return bias;
}

void Neuron::setLearningRate(double lr) {
	learningRate = lr;
}

double Neuron::getTheta(int idx) {
	if (idx >= theta.size())
		return 0.0;
	else
		return theta[idx];
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

double Neuron::getDerrDout() {
	return derrdout;
}

double Neuron::getDerrDnet() {
	return derrdnet;
}

void Neuron::recalculate() {
	lastNet = 0.0;
	for (int i = 0; i < nodeForInput.size(); i++) {
		if (nodeForInput[i] != nullptr) {
			double inputi = nodeForInput[i]->getOutput();
			lastNet += inputi * theta[i];
		}
	}
	lastNet += bias;
	lastOutput = activator(lastNet);
}

void Neuron::recalculateDerrDnet() {
	derrdout = 0.0;
	for (int i = 0; i < nodeForOutput.size(); i++) {
		NodePtr outi = nodeForOutput[i].first;
		int idx = nodeForOutput[i].second;
		if (outi->getType() == 3) {
			OutputNodePtr o = dynamic_cast<OutputNodePtr>(outi);
			derrdout += o->getDerrDout();
		} else if (outi->getType() == 2) {
			NeuronPtr on = dynamic_cast<NeuronPtr>(outi);
			derrdout += on->getDerrDnet() * on->getTheta(idx);
		}
	}
	derrdnet = derrdout * deactivator(lastNet, lastOutput);
}

void Neuron::train() {
	for (int i = 0; i < theta.size(); i++) {
		double dtheta = derrdnet * nodeForInput[i]->getOutput();
		theta[i] -= learningRate * dtheta;
	}
	bias -= learningRate * derrdnet;
}

void Neuron::setInputSize(int size) {
	inputCount = size;
	int thetaOldSize = theta.size();
	theta.resize(inputCount);
	for (int i = thetaOldSize; i < theta.size(); i++) {
		theta[i] = -1.0 + 2.0 * (double)rand() / (double)RAND_MAX;
	}
	nodeForInput.resize(inputCount, nullptr);
}

NeuralNetwork::NeuralNetwork(int iCount, int oCount, double lr, Activation func, Deactivation dfunc) {
	inputCount = iCount;
	outputCount = oCount;
	learningRate = lr;
	defaultActivationFunction = func;
	defaultDeactivationFunction = dfunc;
	for (int i = 0; i < inputCount; i++) {
		inputs.push_back(new InputNode());
	}
	for (int i = 0; i < outputCount; i++) {
		OutputNodePtr outNode = new OutputNode();
		outputNodes.push_back(outNode);
		NeuronPtr outNeuron = new Neuron(inputCount, defaultActivationFunction, defaultDeactivationFunction);
		outNeuron->name = "o" + to_string(i + 1);
		outNeuron->setLearningRate(lr);
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

void NeuralNetwork::setLearningRate(double lr) {
	learningRate = lr;
	for (int i = 0; i < allNeuron.size(); i++) {
		allNeuron[i]->setLearningRate(learningRate);
	}
}

void NeuralNetwork::setActivationFunction(Activation func, Deactivation dfunc) {
	defaultActivationFunction = func;
	defaultDeactivationFunction = dfunc;
	for (int i = 0; i < allNeuron.size(); i++) {
		allNeuron[i]->setActivationFunction(defaultActivationFunction, defaultDeactivationFunction);
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
	for (int i = 0; i < outputNodes.size(); i++) {
		outputNodes[i]->recalculate();
	}
	for (int i = neuronTraverse.size() - 1; i >= 0; i--) {
		neuronTraverse[i]->recalculateDerrDnet();
	}
}

NeuronPtr NeuralNetwork::addNeuron(string name) {
	NeuronPtr newNeuron = new Neuron(inputCount, defaultActivationFunction, defaultDeactivationFunction);
	newNeuron->name = name;
	for (int i = 0; i < inputCount; i++) {
		newNeuron->setInput(dynamic_cast<NodePtr>(inputs[i]), i);
	}
	allNeuron.push_back(newNeuron);
	return newNeuron;
}

void NeuralNetwork::setInput(const vector<double>& input) {
	for (int i = 0; i < min(input.size(), inputs.size()); i++) {
		inputs[i]->setValue(input[i]);
	}
	feedForward();
}

void NeuralNetwork::setTarget(const vector<int>& target) {
	for (int i = 0; i < min(target.size(), outputNodes.size()); i++) {
		outputNodes[i]->setTarget(target[i]);
	}
	feedForward();
}

void NeuralNetwork::recalculate() {
	traverse();
	feedForward();
}

void NeuralNetwork::train() {
	for (int i = neuronTraverse.size() - 1; i >= 0; i--) {
		neuronTraverse[i]->train();
	}
	feedForward();
}

void NeuralNetwork::resetTheta() {
	for (int i = 0; i < allNeuron.size(); i++) {
		NeuronPtr nowneuron = allNeuron[i];
		for (int j = 0; j < nowneuron->getInputNode().size(); j++) {
			nowneuron->setTheta(j, -1.0 + 2.0 * (double)rand() / (double)RAND_MAX);
		}
	}
}

vector<NeuronPtr> NeuralNetwork::getNeurons() {
	return allNeuron;
}

vector<NeuronPtr> NeuralNetwork::getTraverseNeurons() {
	return neuronTraverse;
}

vector<OutputNodePtr> NeuralNetwork::getOutputNodes() {
	return outputNodes;
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

vector<double> NeuralNetwork::getError() {
	vector<double> ans;
	for (int i = 0; i < outputNodes.size(); i++) {
		ans.push_back(outputNodes[i]->getError());
	}
	return ans;
}

vector<string> NeuralNetwork::getTraversePath() {
	vector<string> ans;
	for (int i = 0; i < neuronTraverse.size(); i++) {
		ans.push_back(neuronTraverse[i]->name);
	}
	return ans;
}