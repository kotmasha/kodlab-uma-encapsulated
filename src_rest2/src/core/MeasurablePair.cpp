#include "MeasurablePair.h"
#include "Measurable.h"

extern int ind(int row, int col);
extern int compi(int x);

MeasurablePair::MeasurablePair(ifstream &file, Measurable *_m_i, Measurable *_m_j)
	:_measurable_i(_m_i), _measurable_j(_m_j) {
	file.read((char *)(&v_w), sizeof(double));
}

MeasurablePair::MeasurablePair(Measurable *_m_i, Measurable *_m_j, double w, bool d)
	:_measurable_i(_m_i), _measurable_j(_m_j){
	v_w = w;
	v_d = d;
}

/*
This function is setting pointers to null
*/
void MeasurablePair::pointers_to_null(){
	_w = NULL;
	_d = NULL;
}

/*
This function is setting the weight matrix pointer
*/
void MeasurablePair::setWeightPointers(double *weights){
	int idx_i = _measurable_i->_idx;
	int idx_j = _measurable_j->_idx;
	_w = weights + ind(idx_i, idx_j);
}

/*
This function is setting the dir matrix pointer
*/
void MeasurablePair::setDirPointers(bool *dirs){
	int idx_i = _measurable_i->_idx;
	int idx_j = _measurable_j->_idx;
	_d = dirs + ind(idx_i, idx_j);
}

/*
This function is copying pointer values to value
*/
void MeasurablePair::pointers_to_values(){
	v_w = *_w;
	v_d = *_d;
}

/*
This function is copying value to pointer value
*/
void MeasurablePair::values_to_pointers(){
	*_w = v_w;
	*_d = v_d;
}

/*
This is the function for saving the measurable pairs
Saving order MUST FOLLOW:
1 measurable_id1(will be saved earlier)
2 measurable_id2(will be saved earlier)
3 value of the weight matrix
No dir matrix is needed as it can be concluded from weight
Input: file ofstream
*/
void MeasurablePair::save_measurable_pair(ofstream &file){
	file.write(reinterpret_cast<const char *>(&v_w), sizeof(double));
}

void MeasurablePair::copy_data(MeasurablePair *mp) {
	//dir value will not be copied
	v_w = mp->v_w;
}

double MeasurablePair::getW() {
	return *_w;
}

bool MeasurablePair::getD() {
	return *_d;
}

void MeasurablePair::setW(double &w) {
	*_w = w;
}

void MeasurablePair::setD(bool &d) {
	*_d = d;
}

MeasurablePair::~MeasurablePair(){
	pointers_to_null();
}