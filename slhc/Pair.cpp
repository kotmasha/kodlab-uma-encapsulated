#include "Pair.h"

extern int ind(int row, int col);

Pair::Pair(int idx_y, int idx_x, double &vdist) {
	_idx_y = idx_y;
	_idx_x = idx_x;
	init_values(vdist);
}

void Pair::init_values(double &vdist) {
	_vdist = vdist;
	_vslhc = 0.0;
	_dist = NULL;
	_slhc = NULL;
}

void Pair::pointers_to_values() {
	_vdist = *_dist;
	_vslhc = *_slhc;
}

void Pair::values_to_pointers() {
	*_dist = _vdist;
	*_slhc = _vslhc;
}

void Pair::setPairPointers(double *h_dists, double *h_slhc) {
	_dist = h_dists + ind(_idx_y, _idx_x);
	_slhc = h_slhc + ind(_idx_y, _idx_x);
}

Pair::~Pair() {
	_dist = NULL;
	_slhc = NULL;
}