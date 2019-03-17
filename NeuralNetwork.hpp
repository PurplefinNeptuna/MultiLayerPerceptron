#include <functional>
#include <string>
#include <vector>
using namespace std;

class Node;
class InputNode;
class OutputNode;
class Neuron;

using Activation = function<double(double)>;
using Deactivation = function<double(double, double)>;
using NodePtr = Node*;
using InputNodePtr = InputNode*;
using OutputNodePtr = OutputNode*;
using NeuronPtr = Neuron*;

class Node {
public:
	virtual double getOutput();
	virtual int getType();
};

class InputNode : public Node {
private:
	double value;

public:
	InputNode();
	void setValue(double n);
	double getOutput();
	int getType();
};

class OutputNode : public Node {
private:
	NeuronPtr outNeuron;
	int target;
	double derrdout;

public:
	OutputNode();
	void setOutputNeuron(NeuronPtr neuron);
	void setTarget(int tar);
	void recalculate();
	int getTarget();
	double getDerrDout();
	double getError();
	double getOutput();
	int getPrediction();
	int getType();
};

class Neuron : public Node {
private:
	int inputCount;
	double bias, lastNet, lastOutput, derrdout, derrdnet, learningRate;
	vector<double> theta;
	vector<NodePtr> nodeForInput;
	vector<pair<NodePtr, int>> nodeForOutput;
	Activation activator;
	Deactivation deactivator;

public:
	bool traverseVisited;
	string name;

	Neuron(int iCount, Activation func, Deactivation dfunc);
	void setActivationFunction(Activation func, Deactivation dfunc);
	void setTheta(int idx, double newtheta);
	void setBias(double newbias);
	double getBias();
	void setLearningRate(double lr);
	double getTheta(int idx);
	double getNet();
	double getOutput();
	double getDerrDout();
	double getDerrDnet();
	int getType();
	void setInput(NodePtr input, int idx);
	void addOutput(NodePtr output, int inputNum);
	void removeOutput(NodePtr output);
	vector<NodePtr> getInputNode();
	vector<NodePtr> getOutputNode();
	void recalculate();
	void recalculateDerrDnet();
	void train();
	void setInputSize(int size);
};

class NeuralNetwork {
private:
	int outputCount;
	int inputCount;
	double learningRate;
	Activation defaultActivationFunction;
	Deactivation defaultDeactivationFunction;
	vector<OutputNodePtr> outputNodes;
	vector<NeuronPtr> allNeuron;
	vector<NeuronPtr> neuronTraverse;
	vector<InputNodePtr> inputs;

	void traverse();
	void feedForward();

public:
	NeuralNetwork(int iCount, int oCount, double lr, Activation func, Deactivation dfunc);
	~NeuralNetwork();
	void setLearningRate(double lr);
	void setActivationFunction(Activation func, Deactivation dfunc);
	NeuronPtr addNeuron(string name);
	void setInput(const vector<double>& input);
	void setTarget(const vector<int>& target);
	void recalculate();
	void train();
	void resetTheta();
	vector<NeuronPtr> getNeurons();
	vector<NeuronPtr> getTraverseNeurons();
	vector<OutputNodePtr> getOutputNodes();
	vector<int> getPrediction();
	vector<double> getOutput();
	vector<double> getError();
	vector<string> getTraversePath();
};