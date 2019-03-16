#ifndef G_HELPER
#define G_HELPER
#include "lib/gnuplot_i.hpp"
#include <string>
#include <tuple>
#include <utility>
#include <vector>
using namespace std;

inline void pngGraph(string name, const vector<double>& x, const vector<double>& y, bool logx, bool logy, double xmin, double xmax, double ymin, double ymax, string title, string additionalLineStyle, string color, string lineName = "") {
	if (lineName == "")
		lineName = name;
	Gnuplot* plotpng = new Gnuplot();
	plotpng->cmd("set term png\nset output \"" + name + ".png\"");
	plotpng->set_style("lines " + additionalLineStyle + " lc rgb \"" + color + "\"");
	plotpng->set_title(title);
	plotpng->set_grid();
	if (logx == true) {
		plotpng->set_xlogscale();
	} else {
		plotpng->set_xrange(xmin, xmax);
	}

	if (logy == true) {
		plotpng->set_ylogscale();
	} else {
		plotpng->set_yrange(ymin, ymax);
	}

	plotpng->plot_xy(x, y, lineName);
	delete plotpng;
}

inline void buildGraph(string name, const vector<double>& x, const vector<double>& y, bool logx, bool logy, double xmin, double xmax, double ymin, double ymax, string title, string additionalLineStyle, string color) {
	Gnuplot* plot = new Gnuplot();
	plot->cmd("set output");
	plot->cmd("set term qt title \"" + title + "\"");
	plot->set_style("lines " + additionalLineStyle + " lc rgb \"" + color + "\"");
	plot->set_title(title);
	plot->set_grid();
	if (logx == true) {
		plot->set_xlogscale();
	} else {
		plot->set_xrange(xmin, xmax);
	}

	if (logy == true) {
		plot->set_ylogscale();
	} else {
		plot->set_yrange(ymin, ymax);
	}

	plot->plot_xy(x, y, name);

	pngGraph(name, x, y, logx, logy, xmin, xmax, ymin, ymax, title, additionalLineStyle, color);
}

#endif