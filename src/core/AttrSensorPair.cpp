#include "AttrSensorPair.h"
#include "AttrSensor.h"
#include "UMAException.h"
#include "Logger.h"

extern int ind(int row, int col);
extern int compi(int x);

static Logger attrSensorPairLogger("AttrSensorPair", "log/attrSensor.log");

/*
AttrSensorPair::AttrSensorPair(ifstream &file, AttrSensor *_m_i, AttrSensor *_m_j)
	:_measurable_i(_m_i), _measurable_j(_m_j) {
	file.read((char *)(&v_w), sizeof(double));
}
*/

AttrSensorPair::AttrSensorPair(AttrSensor * const _attrSensorI, AttrSensor * const _attrSensorJ, double w, bool d)
	:_attrSensorI(_attrSensorI), _attrSensorJ(_attrSensorJ){
	_vw = w;
	_vd = d;
	attrSensorPairLogger.debug("A new attr sensor pair is created with, attrSensor1=" + to_string(_attrSensorI->getIdx()) +
		", attrSensor2=" + to_string(_attrSensorJ->getIdx()));
}

/*
This function is setting pointers to null
*/
void AttrSensorPair::pointersToNull(){
	_w = NULL;
	_d = NULL;
}

/*
This function is setting the weight matrix pointer
*/
void AttrSensorPair::setWeightPointers(double *weights){
	int idx_i = _attrSensorI->_idx;
	int idx_j = _attrSensorJ->_idx;
	_w = weights + ind(idx_i, idx_j);
}

/*
This function is setting the dir matrix pointer
*/
void AttrSensorPair::setDirPointers(bool *dirs){
	int idx_i = _attrSensorI->_idx;
	int idx_j = _attrSensorJ->_idx;
	_d = dirs + ind(idx_i, idx_j);
}

/*
This function is copying pointer values to value
*/
void AttrSensorPair::pointersToValues(){
	if (!_w || !_d) {
		throw UMAException("The weights or dirs pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	_vw = *_w;
	_vd = *_d;
	attrSensorPairLogger.debug("Pointer values copied to object values attrSensor1 = " + to_string(_attrSensorI->getIdx()) +
		", attrSensor2=" + to_string(_attrSensorJ->getIdx()));
}

/*
This function is copying value to pointer value
*/
void AttrSensorPair::valuesToPointers(){
	if (!_w || !_d) {
		throw UMAException("The weights or dirs pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	*_w = _vw;
	*_d = _vd;
	attrSensorPairLogger.debug("Pointer object values copied to sensor values attrSensor1 = " + to_string(_attrSensorI->getIdx()) +
		", attrSensor2=" + to_string(_attrSensorJ->getIdx()));
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
	if (!_w) {
		throw UMAException("The weights pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_w;
}

const bool &AttrSensorPair::getD() const{
	if (!_d) {
		throw UMAException("The dirs pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_d;
}

void AttrSensorPair::setW(const double w) {
	if (!_w) {
		throw UMAException("The weights pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	*_w = w;
}

void AttrSensorPair::setD(const bool d) {
	if (!_d) {
		throw UMAException("The dirs pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	*_d = d;
}

AttrSensorPair::~AttrSensorPair(){
	pointersToNull();
}