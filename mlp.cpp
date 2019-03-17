/*
C++ to GNUPlot (gnuplot_i.hpp) source code:
https://code.google.com/archive/p/gnuplot-cpp/

CSV reader (csv.h) source code:
https://github.com/ben-strasser/fast-cpp-csv-parser/
*/

#include "NeuralNetwork.hpp"
#include "graph_helper.hpp"
#include "lib/csv.h"
#include <bits/stdc++.h>
using namespace std;

using vvdvi = vector<pair<vector<double>, vector<int>>>;

pair<double, double> runEpoch(NeuralNetwork& nn, int trainSize, int validSize);
void runMLP(double lrate, int epoch, int trainSize, int validSize, NeuralNetwork& nn);

vvdvi csvdata;

int main() {
	srand(time(nullptr));
	double x1, x2, x3, x4;
	int t1, t2;
	io::CSVReader<6> csv("iris_mlp.csv");
	csv.read_header(io::ignore_extra_column, "x1", "x2", "x3", "x4", "t1", "t2");
	while (csv.read_row(x1, x2, x3, x4, t1, t2)) {
		vector<double> input = { x1, x2, x3, x4 };
		vector<int> output = { t1, t2 };
		csvdata.push_back(make_pair(input, output));
	}

	Activation sigmoid = [](double x) { return 1.0 / (1 + exp(-x)); };
	Deactivation dsigmoid = [](double x, double y) { return y * (1.0 - y); };
	NeuralNetwork nn(4, 2, 0.1, sigmoid, dsigmoid);
	vector<NeuronPtr> outNeuron = nn.getNeurons();
	vector<NeuronPtr> hiddenNeuron;
	hiddenNeuron.push_back(nn.addNeuron("h1"));
	hiddenNeuron.push_back(nn.addNeuron("h2"));
	hiddenNeuron.push_back(nn.addNeuron("h3"));
	hiddenNeuron.push_back(nn.addNeuron("h4"));
	for (int i = 0; i < outNeuron.size(); i++) {
		outNeuron[i]->setBias(-1.0 + 2.0 * (double)rand() / (double)RAND_MAX);
		for (int j = 0; j < hiddenNeuron.size(); j++) {
			NodePtr childPtr = dynamic_cast<NodePtr>(hiddenNeuron[j]);
			NodePtr parentPtr = dynamic_cast<NodePtr>(outNeuron[i]);
			outNeuron[i]->setInput(childPtr, j);
			hiddenNeuron[j]->addOutput(parentPtr, j);
		}
	}
	for (int i = 0; i < hiddenNeuron.size(); i++) {
		hiddenNeuron[i]->setBias(-1.0 + 2.0 * (double)rand() / (double)RAND_MAX);
	}
	nn.recalculate();

	printf("x1,x2,x3,x4,t1,t2,,");
	vector<NeuronPtr> allN = nn.getTraverseNeurons();
	for (int i = 0; i < allN.size(); i++) {
		NeuronPtr nnow = allN[i];
		vector<NodePtr> nnowInput = nnow->getInputNode();
		for (int j = 0; j < nnowInput.size(); j++) {
			printf("%s,", (nnow->name + "w" + to_string(j + 1)).c_str());
		}
		const char* cname = nnow->name.c_str();
		printf("%sb,dEdNet%s,dEdOut%s,%snet,%sout,,", cname, cname, cname, cname, cname);
	}
	vector<OutputNodePtr> allO = nn.getOutputNodes();
	for (int i = 0; i < allO.size(); i++) {
		printf("target%d,output%d,error%d,,", i + 1, i + 1, i + 1);
	}
	printf("\n");

	runMLP(0.1, 1000, 120, 30, nn);
	runMLP(0.8, 1000, 120, 30, nn);

	getchar();
}

void runMLP(double lrate, int epoch, int trainSize, int validSize, NeuralNetwork& nn) {
	char cname[255];
	vector<double> errx, erry, accx, accy;
	pair<double, double> eaEpoch;

	sprintf(cname, "%.1lf", lrate);
	string lrateName = string(cname);

	nn.resetTheta();
	nn.setLearningRate(lrate);

	for (int i = 0; i < epoch; i++) {
		errx.push_back(i);
		accx.push_back(i);
		eaEpoch = runEpoch(nn, trainSize, validSize);
		erry.push_back(eaEpoch.first);
		accy.push_back(eaEpoch.second);
	}
	string errName = "error " + lrateName, errTitle = "Error/epoch (lr=" + lrateName + ")", errStyle = "lw 3", errColor = "#FF0000";
	buildGraph(errName, errx, erry, false, true, -10, epoch + 10, 0, 0, errTitle, errStyle, errColor);

	string accName = "accuracy " + lrateName, accTitle = "Accuracy/epoch (lr=" + lrateName + ")", accStyle = "lw 3", accColor = "#0000FF";
	buildGraph(accName, accx, accy, false, false, -10, epoch + 10, 0, 1.05, accTitle, accStyle, accColor);
}

pair<double, double> runEpoch(NeuralNetwork& nn, int trainSize, int validSize) {
	pair<vector<double>, vector<int>> data1;
	vector<double> err;
	vector<int> pred;
	double errtot = 0.0;
	double acctot = 0.0;

	for (int i = 0; i < trainSize; i++) {
		data1 = csvdata[i];

		nn.setInput(data1.first);
		nn.setTarget(data1.second);
		err = nn.getError();

		printf("%.1f,%.1f,%.1f,%.1f,%d,%d,,", data1.first[0], data1.first[1], data1.first[2], data1.first[3], data1.second[0], data1.second[1]);
		vector<NeuronPtr> allN = nn.getTraverseNeurons();
		for (int i = 0; i < allN.size(); i++) {
			NeuronPtr nnow = allN[i];
			vector<NodePtr> nnowInput = nnow->getInputNode();
			for (int j = 0; j < nnowInput.size(); j++) {
				printf("%lf,", nnow->getTheta(j));
			}
			printf("%lf,%lf,%lf,%lf,%lf,,", nnow->getBias(), nnow->getDerrDnet(), nnow->getDerrDout(), nnow->getNet(), nnow->getOutput());
		}
		vector<OutputNodePtr> allO = nn.getOutputNodes();
		for (int i = 0; i < allO.size(); i++) {
			printf("%d,%lf,%lf,,", allO[i]->getTarget(), allO[i]->getOutput(), allO[i]->getError());
		}
		printf("\n");

		nn.train();

		double errnow = 0.0;
		for (int j = 0; j < err.size(); j++) {
			errnow += err[j];
		}
		errtot += errnow;
	}

	for (int i = trainSize; i < trainSize + validSize; i++) {
		data1 = csvdata[i];

		nn.setInput(data1.first);
		nn.setTarget(data1.second);
		pred = nn.getPrediction();

		if (pred[0] == data1.second[0] && pred[1] == data1.second[1]) {
			acctot += 1.0;
		}
	}

	return make_pair(errtot / trainSize, acctot / validSize);
}