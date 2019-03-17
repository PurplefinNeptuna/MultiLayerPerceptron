#include <functional>
#include <vector>
using namespace std;

class Node;
class InputNode;
class OutputNode;
class Neuron;

using Activation = function<double(double)>;
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

public:
	OutputNode();
	void setOutputNeuron(NeuronPtr neuron);
	double getOutput();
	int getPrediction();
	int getType();
};

class Neuron : public Node {
private:
	int inputCount;
	double bias, lastNet, lastOutput;
	vector<double> theta;
	vector<NodePtr> nodeForInput;
	vector<pair<NodePtr, int>> nodeForOutput;
	Activation activator;

public:
	bool traverseVisited;

	Neuron(int iCount, Activation func);
	void setActivationFunction(Activation func);
	double getNet();
	double getOutput();
	int getType();
	void setInput(NodePtr input, int idx);
	void addOutput(NodePtr output, int inputNum);
	void removeOutput(NodePtr output);
	vector<NodePtr> getInputNode();
	vector<NodePtr> getOutputNode();
	vector<pair<NodePtr, int>> getOutputNodeWithInputIndex();
	void recalculate();
	void setInputSize(int size);
};

class NeuralNetwork {
private:
	int outputCount;
	int inputCount;
	Activation defaultActivationFunction;
	vector<OutputNodePtr> outputNodes;
	vector<NeuronPtr> allNeuron;
	vector<NeuronPtr> neuronTraverse;
	vector<InputNodePtr> inputs;

	void traverse();
	void feedForward();

public:
	NeuralNetwork(int iCount, int oCount, Activation func);
	~NeuralNetwork();
	void setActivationFunction(Activation func);
	NeuronPtr addNeuron();
	void setInput(const vector<double>& input);
	void recalculate();
	vector<NeuronPtr> getNeurons();
	vector<int> getPrediction();
	vector<double> getOutput();
};