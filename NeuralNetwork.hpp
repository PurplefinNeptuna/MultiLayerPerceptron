#include <functional>
#include <vector>
using namespace std;

class Node;
class InputNode;
class Neuron;

using Activation = function<double(double)>;
using NodePtr = Node*;
using InputNodePtr = InputNode*;
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

class Neuron : public Node {
private:
	int inputCount;
	double bias, lastNet, lastOutput;
	vector<double> theta;
	vector<NodePtr> nodeForInput;
	Activation activator;

public:
	bool traverseVisited;

	Neuron(int iCount);
	Neuron(int iCount, Activation func);
	void setActivationFunction(Activation func);
	double getNet();
	double getOutput();
	int getType();
	void setInput(NodePtr input, int idx);
	vector<NodePtr> getInputNode();
	void recalculate();
	void setInputSize(int size);
};

class NeuralNetwork {
private:
	int outputCount;
	int inputCount;
	Activation defaultActivationFunction;
	vector<NeuronPtr> outputNeurons;
	vector<NeuronPtr> hiddenNeurons;
	vector<NeuronPtr> neuronTraverse;
	vector<InputNodePtr> inputs;

	void traverse();
	void feedForward();

public:
	NeuralNetwork(int iCount, int oCount);
	NeuralNetwork(int iCount, int oCount, Activation func);
	~NeuralNetwork();
	void setActivationFunction(Activation func);
	NeuronPtr addHiddenNeuron();
	void setInput(const vector<double>& input);
	void recalculate();
	vector<NeuronPtr> getOutputNeurons();
	vector<NeuronPtr> getHiddenNeurons();
	vector<int> getPrediction();
	vector<double> getOutput();
};