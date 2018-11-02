#include "AttrSensor.h"
#include "UMAException.h"
#include "Logger.h"

extern int ind(int row, int col);
extern int compi(int x);

static Logger attrSensorLogger("AttrSensor", "log/attrSensor.log");

/*
AttrSensor::AttrSensor(ifstream &file) {
	int uuid_length = -1;
	file.read((char *)(&uuid_length), sizeof(int));
	if (uuid_length > 0) {
		_uuid = string(uuid_length, ' ');
		file.read(&_uuid[0], uuid_length * sizeof(char));
	}
	else _uuid = "";
	file.read((char *)&_idx, sizeof(int));
	file.read((char *)&_isOriginPure, sizeof(bool));
	pointersToNull();
}
*/

AttrSensor::AttrSensor(const string &uuid, UMACoreObject *parent, int idx, bool isOriginPure, double diag)
	: UMACoreObject(uuid, UMA_OBJECT::ATTR_SENSOR, parent){
	_idx = idx;
	_isOriginPure = isOriginPure;
	pointersToNull();
	_vdiag = diag;
	_vdiag_ = diag;
	_vobserve = false;
	_vobserve_ = false;
	_vtarget = false;

	attrSensorLogger.debug("A new attr sensor is created with id=" + uuid, this->getParentChain());
}

/*
The function is setting all pointers to NULL
*/
void AttrSensor::pointersToNull(){
	_diag = NULL;
	_diag_ = NULL;
	_observe = NULL;
	_observe_ = NULL;
	_current = NULL;
	_target = NULL;
	_prediction = NULL;
}

/*
The function is copying the pointer value to values in the object
*/
void AttrSensor::pointersToValues(){
	if (!_diag || !_diag_ || !_observe || !_observe_ || !_target) {
		throw UMABadOperationException("The diag or observe or target pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	_vdiag = *_diag;
	_vdiag_ = *_diag_;
	_vobserve = *_observe;
	_vobserve_ = *_observe_;
	_vtarget = *_target;

	attrSensorLogger.debug("Pointer values copied to object values for attr sensor=" + _uuid, this->getParentChain());
}

/*
The function is copying values to pointer in the object
*/
void AttrSensor::valuesToPointers(){
	if (!_diag || !_diag_ || !_observe || !_observe_ || !_target) {
		throw UMABadOperationException("The diag or observe or target pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	*_diag = _vdiag;
	*_diag_ = _vdiag_;
	*_observe = _vobserve;
	*_observe_ = _vobserve_;
	*_target = _vtarget;

	attrSensorLogger.debug("Object values copied to pointer values for attr sensor=" + _uuid, this->getParentChain());
}

/*
This is setting the diagonal value pointer
*/
void AttrSensor::setDiagPointers(double *_diags, double *_diags_){
	_diag = _diags + _idx;
	_diag_ = _diags_ + _idx;
}

/*
This is setting the observe/observe_ value pointer
*/
void AttrSensor::setObservePointers(bool *observe, bool *observe_){
	_observe = observe + _idx;
	_observe_ = observe_ + _idx;
}

/*
This is setting the current value pointer
*/
void AttrSensor::setCurrentPointers(bool *current, bool *current_) {
	_current = current + _idx;
}

/*
This is setting the current value pointer
*/
void AttrSensor::setTargetPointers(bool *target) {
	_target = target + _idx;
}

/*
This is setting the prediction value pointer
*/
void AttrSensor::setPredictionPointers(bool *prediction) {
	_prediction = prediction + _idx;
}

/*
This function is setting the measurable idx
Input: new idx
*/
void AttrSensor::setIdx(int idx){
	_idx = idx;
}

void AttrSensor::setObserve(bool observe) {
	if (!_observe) {
		throw UMABadOperationException("The observe pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	*_observe = observe;
}

void AttrSensor::setOldObserve(bool observe_) {
	if (!_observe_) {
		throw UMABadOperationException("The observe_ pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	*_observe_ = observe_;
}

/*
This function is save measurable
Saving order MUST FOLLOW:
1 measurable idx
2 whether the measurable is originally pure
*/
void AttrSensor::saveAttrSensor(ofstream &file){
	int uuidLength = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuidLength), sizeof(int));
	file.write(_uuid.c_str(), uuidLength * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_isOriginPure), sizeof(bool));
	file.write(reinterpret_cast<const char *>(&_vdiag), sizeof(double));
}

AttrSensor *AttrSensor::loadAttrSensor(ifstream &file, UMACoreObject *parent) {
	int uuidLength = -1;
	string uuid;
	int idx = -1;
	double diag = 0.0;
	bool isOriginPure = false;
	file.read((char *)(&uuidLength), sizeof(int));
	if (uuidLength > 0) {
		uuid = string(uuidLength, ' ');
		file.read(&uuid[0], uuidLength * sizeof(char));
	}
	else uuid = "";
	file.read((char *)&idx, sizeof(int));
	file.read((char *)&isOriginPure, sizeof(bool));
	file.read((char *)&diag, sizeof(double));

	AttrSensor *attrSensor = new AttrSensor(uuid, parent, idx, isOriginPure, diag);

	return attrSensor;
}

/*
void AttrSensor::copy_data(AttrSensor *m) {
	_isOriginPure = m->_isOriginPure;
	_vdiag = m->_vdiag;
	_vdiag_ = m->_vdiag_;
}
*/

const double &AttrSensor::getDiag() {
	if (!_diag) {
		throw UMABadOperationException("The diag pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	return *_diag;
}

const double &AttrSensor::getOldDiag() {
	if (!_diag_) {
		throw UMABadOperationException("The diag_ pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	return *_diag_;
}

const bool &AttrSensor::getIsOriginPure(){
	return _isOriginPure;
}

const bool &AttrSensor::getObserve(){
	if (!_observe) {
		throw UMABadOperationException("The observe pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	return *_observe;
}

const bool &AttrSensor::getTarget() {
	if (!_target) {
		throw UMABadOperationException("The target pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	return *_target;
}

const bool &AttrSensor::getOldObserve() {
	if (!_observe_) {
		throw UMABadOperationException("The observe_ pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	return *_observe_;
}

const bool &AttrSensor::getCurrent() {
	if (!_current) {
		throw UMABadOperationException("The current pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	return *_current;
}

const int &AttrSensor::getIdx() const{
	return _idx;
}

void AttrSensor::setDiag(const double &diag) {
	if (!_diag) {
		throw UMABadOperationException("The diag pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	*_diag = diag;
}

void AttrSensor::setOldDiag(const double &diag_) {
	if (!_diag_) {
		throw UMABadOperationException("The diag_ pointer is not initiated!", false, &attrSensorLogger, this->getParentChain());
	}
	*_diag_ = diag_;
}

void AttrSensor::setIsOriginPure(const bool &isOriginPure) {
	_isOriginPure = isOriginPure;
}

AttrSensor::~AttrSensor(){
	pointersToNull();
}