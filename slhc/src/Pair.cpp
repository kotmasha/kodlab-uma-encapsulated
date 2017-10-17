#include "Pair.h"

extern int ind(int row, int col);

Pair::Pair(int idx_y, int idx_x, double &vdist) {
	_idx_y = idx_y;
	_idx_x = idx_x;
	init_values(vdist);
}

void Pair::init_values(double &vdist) {
	// init dist as input, slhc to 0, pointers to NULL
	_vdist = vdist;
	_vslhc = 0.0;
	_dist = NULL;
	_slhc = NULL;
}

void Pair::pointers_to_values() {
	// copy pointer value to value for store
	_vdist = *_dist;
	_vslhc = *_slhc;
}

void Pair::values_to_pointers() {
	// copy value to pointer value
	*_dist = _vdist;
	*_slhc = _vslhc;
}

void Pair::setPairPointers(double *h_dists, double *h_slhc) {
	// setup the idx from pointer to array
	_dist = h_dists + ind(_idx_y, _idx_x);
	_slhc = h_slhc + ind(_idx_y, _idx_x);
}

void Pair::setIdx(int idx_y, int idx_x) {
	// set the current idx
	_idx_y = idx_y;
	_idx_x = idx_x;
}

Pair::~Pair() {
	_dist = NULL;
	_slhc = NULL;
}