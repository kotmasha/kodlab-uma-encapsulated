#ifndef _PAIR_
#define _PAIR_

/*
The pair object
*/

#include "Header.h"

class Pair {
public:
	//x and y idx
	int _idx_x, _idx_y;
	// dist value of the point(storing for tmp purpose), and the pointer pointing to the location in dist matrix
	double _vdist, *_dist;
	// slhc value of the point(storing for tmp purpose), and the pointer pointing to the location in slhc matrix
	double _vslhc, *_slhc;

public:
	Pair(int idx_y, int idx_x, double &vdist);
	void init_values(double &vdist);
	void pointers_to_values();
	void values_to_pointers();
	void setPairPointers(double *h_dists, double *h_slhc);
	void setIdx(int idx_y, int idx_x);
	~Pair();
};

#endif