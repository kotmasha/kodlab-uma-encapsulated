#include "AttrSensorPair.h"
#include "AttrSensor.h"

extern int ind(int row, int col);
extern int compi(int x);

/*
AttrSensorPair::AttrSensorPair(ifstream &file, AttrSensor *_m_i, AttrSensor *_m_j)
	:_measurable_i(_m_i), _measurable_j(_m_j) {
	file.read((char *)(&v_w), sizeof(double));
}
*/

AttrSensorPair::AttrSensorPair(AttrSensor * const _m_i, AttrSensor * const _m_j, double w, bool d)
	:_measurable_i(_m_i), _measurable_j(_m_j){
	v_w = w;
	v_d = d;
}

/*
This function is setting pointers to null
*/
void AttrSensorPair::pointers_to_null(){
	_w = NULL;
	_d = NULL;
}

/*
This function is setting the weight matrix pointer
*/
void AttrSensorPair::setWeightPointers(double *weights){
	int idx_i = _measurable_i->_idx;
	int idx_j = _measurable_j->_idx;
	_w = weights + ind(idx_i, idx_j);
}

/*
This function is setting the dir matrix pointer
*/
void AttrSensorPair::setDirPointers(bool *dirs){
	int idx_i = _measurable_i->_idx;
	int idx_j = _measurable_j->_idx;
	_d = dirs + ind(idx_i, idx_j);
}

/*
This function is copying pointer values to value
*/
void AttrSensorPair::pointers_to_values(){
	v_w = *_w;
	v_d = *_d;
}

/*
This function is copying value to pointer value
*/
void AttrSensorPair::values_to_pointers(){
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
/*
void AttrSensorPair::save_measurable_pair(ofstream &file){
	file.write(reinterpret_cast<const char *>(&v_w), sizeof(double));
}

void AttrSensorPair::copy_data(AttrSensorPair *mp) {
	//dir value will not be copied
	v_w = mp->v_w;
}
*/

const double &AttrSensorPair::getW() const{
	return *_w;
}

const bool &AttrSensorPair::getD() const{
	return *_d;
}

void AttrSensorPair::setW(const double w) {
	*_w = w;
}

void AttrSensorPair::setD(const bool d) {
	*_d = d;
}

AttrSensorPair::~AttrSensorPair(){
	pointers_to_null();
}