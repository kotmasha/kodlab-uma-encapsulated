#include "AttrSensorPair.h"
#include "AttrSensor.h"
#include "UMAException.h"
#include "Logger.h"

extern int ind(int row, int col);
extern int compi(int x);

static Logger attrSensorPairLogger("AttrSensorPair", "log/attrSensor.log");

AttrSensorPair::AttrSensorPair(UMACoreObject *parent, AttrSensor * const attrSensorI, AttrSensor * const attrSensorJ, double w, bool d)
	: UMACoreObject(generateUUID(attrSensorI, attrSensorJ), UMA_OBJECT::ATTR_SENSOR_PAIR, parent)
	,_attrSensorI(attrSensorI), _attrSensorJ(attrSensorJ){
	_vw = w;
	_vd = d;
	attrSensorPairLogger.debug("A new attr sensor pair is created with, attrSensor1=" + to_string(_attrSensorI->getIdx()) +
		", attrSensor2=" + to_string(_attrSensorJ->getIdx()), this->getParentChain());
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
		throw UMABadOperationException("The weights or dirs pointer is not initiated!", false, &attrSensorPairLogger, this->getParentChain());
	}
	_vw = *_w;
	_vd = *_d;
	attrSensorPairLogger.debug("Pointer values copied to object values attrSensor1 = " + to_string(_attrSensorI->getIdx()) +
		", attrSensor2=" + to_string(_attrSensorJ->getIdx()), this->getParentChain());
}

/*
This function is copying value to pointer value
*/
void AttrSensorPair::valuesToPointers(){
	if (!_w || !_d) {
		throw UMABadOperationException("The weights or dirs pointer is not initiated!", false, &attrSensorPairLogger, this->getParentChain());
	}
	*_w = _vw;
	*_d = _vd;
	attrSensorPairLogger.debug("Pointer object values copied to sensor values attrSensor1 = " + to_string(_attrSensorI->getIdx()) +
		", attrSensor2=" + to_string(_attrSensorJ->getIdx()), this->getParentChain());
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

void AttrSensorPair::saveAttrSensorPair(ofstream &file){
	file.write(reinterpret_cast<const char *>(&_vw), sizeof(double));
}

AttrSensorPair *AttrSensorPair::loadAttrSensorPair(ifstream &file, AttrSensor *attrSensorI, AttrSensor *attrSensorJ,
	bool b, UMACoreObject *parent) {
	double vw = 0.0;
	file.read((char *)(&vw), sizeof(double));

	AttrSensorPair *attrSensorPair = new AttrSensorPair(parent, attrSensorI, attrSensorJ, vw, b);

	return attrSensorPair;
}

/*
This function is copy data from attrSensorPair to current attrSensorPair
*/
void AttrSensorPair::mergeAttrSensorPair(AttrSensorPair * const attrSensorPair){
	//dir value will not be copied
	_vw = attrSensorPair->_vw;
}

const double &AttrSensorPair::getW(){
	if (!_w) {
		throw UMABadOperationException("The weights pointer is not initiated!", false, &attrSensorPairLogger, this->getParentChain());
	}
	return *_w;
}

const bool &AttrSensorPair::getD(){
	if (!_d) {
		throw UMABadOperationException("The dirs pointer is not initiated!", false, &attrSensorPairLogger, this->getParentChain());
	}
	return *_d;
}

void AttrSensorPair::setW(const double w) {
	if (!_w) {
		throw UMABadOperationException("The weights pointer is not initiated!", false, &attrSensorPairLogger, this->getParentChain());
	}
	*_w = w;
}

void AttrSensorPair::setD(const bool d) {
	if (!_d) {
		throw UMABadOperationException("The dirs pointer is not initiated!", false, &attrSensorPairLogger, this->getParentChain());
	}
	*_d = d;
}

AttrSensorPair::~AttrSensorPair(){
	pointersToNull();
}

const string AttrSensorPair::generateUUID(AttrSensor * const _attrSensorI, AttrSensor * const _attrSensorJ) const {
	return _attrSensorI->_uuid + "-" + _attrSensorJ->_uuid;
}