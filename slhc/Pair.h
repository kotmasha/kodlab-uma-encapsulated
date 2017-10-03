#ifndef _PAIR_
#define _PAIR_

#include "Header.h"

class Pair {
protected:
	int _idx_x, _idx_y;
	double _vdist, *_dist;
	double _vslhc, *_slhc;

public:
	Pair(int idx_y, int idx_x, double &vdist);
	void init_values(double &vdist);
	void pointers_to_values();
	void values_to_pointers();
	void setPairPointers(double *h_dists, double *h_slhc);
	~Pair();
};

#endif