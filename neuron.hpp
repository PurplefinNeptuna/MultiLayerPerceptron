using namespace std;

class Neuron {
private:
	double bias;

public:
	Neuron(double bias = 0.0);
	void setBias(double bias);
	double getBias();
};