#include "Measurable.h"

extern int ind(int row, int col);
extern int compi(int x);

Measurable::Measurable(ifstream &file) {
	file.read((char *)&_idx, sizeof(int));
	file.read((char *)&_isOriginPure, sizeof(bool));
	pointers_to_null();
}

Measurable::Measurable(int idx, bool isOriginPure){
	_idx = idx;
	_isOriginPure = isOriginPure;
	pointers_to_null();
}

/*
The function is setting all pointers to NULL
*/
void Measurable::pointers_to_null(){
	_diag = NULL;
	_diag_ = NULL;
	_status = NULL;
}

/*
The function is copying the pointer value to values in the object
*/
void Measurable::pointers_to_values(){
	_vdiag = *_diag;
	_vdiag_ = *_diag_;
}

/*
The function is copying values to pointer in the object
*/
void Measurable::values_to_pointers(){
	*_diag = _vdiag;
	*_diag_ = _vdiag_;
}

/*
This is setting the diagonal value pointer
*/
void Measurable::setDiagPointers(double *_diags, double *_diags_){
	_diag = _diags + _idx;
	_diag_ = _diags_ + _idx;
}

/*
This is setting the status/current value pointer
*/
void Measurable::setStatusPointers(bool *status){
	_status = status + _idx;
}

/*
This function is setting the measurable idx
Input: new idx
*/
void Measurable::setIdx(int idx){
	_idx = idx;
}

/*
This function is save measurable
Saving order MUST FOLLOW:
1 measurable idx
2 whether the measurable is originally pure
*/
void Measurable::save_measurable(ofstream &file){
	file.write(reinterpret_cast<const char *>(&_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_isOriginPure), sizeof(bool));
}

Measurable::~Measurable(){
	pointers_to_null();
}