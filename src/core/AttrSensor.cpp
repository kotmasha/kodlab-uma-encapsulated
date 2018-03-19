#include "AttrSensor.h"
#include "UMAException.h"
#include "Logger.h"

extern int ind(int row, int col);
extern int compi(int x);

static Logger attrSensorLogger("attrSensor", "log/attrSensor.log");

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
	pointers_to_null();
}
*/

AttrSensor::AttrSensor(const string &uuid, int idx, bool isOriginPure, double diag): _uuid(uuid){
	_idx = idx;
	_isOriginPure = isOriginPure;
	pointers_to_null();
	_vdiag = diag;
	_vdiag_ = diag;
	_vobserve = false;
	_vobserve_ = false;

	//attrSensorLogger.debug("A new attr sensor is created with id=" + uuid);
}

/*
The function is setting all pointers to NULL
*/
void AttrSensor::pointers_to_null(){
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
void AttrSensor::pointers_to_values(){
	if (!_diag || !_diag_ || !_observe || !_observe_) {
		throw UMAException("The diag or observe pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	_vdiag = *_diag;
	_vdiag_ = *_diag_;
	_vobserve = *_observe;
	_vobserve_ = *_observe_;

	attrSensorLogger.debug("Pointer values copied to object values for attr sensor=" + _uuid);
}

/*
The function is copying values to pointer in the object
*/
void AttrSensor::values_to_pointers(){
	if (!_diag || !_diag_ || !_observe || !_observe_) {
		throw UMAException("The diag or observe pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	*_diag = _vdiag;
	*_diag_ = _vdiag_;
	*_observe = _vobserve;
	*_observe_ = _vobserve_;

	attrSensorLogger.debug("Object values copied to pointer values for attr sensor=" + _uuid);
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
void AttrSensor::setCurrentPointers(bool *current) {
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

/*
This function is save measurable
Saving order MUST FOLLOW:
1 measurable idx
2 whether the measurable is originally pure
*/
/*
void AttrSensor::save_measurable(ofstream &file){
	int sid_length = _uuid.length();
	file.write(reinterpret_cast<const char *>(&sid_length), sizeof(int));
	file.write(_uuid.c_str(), sid_length * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_isOriginPure), sizeof(bool));
}
*/

/*
void AttrSensor::copy_data(AttrSensor *m) {
	_isOriginPure = m->_isOriginPure;
	_vdiag = m->_vdiag;
	_vdiag_ = m->_vdiag_;
}
*/

const double &AttrSensor::getDiag() const{
	if (!_diag) {
		throw UMAException("The diag pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_diag;
}

const double &AttrSensor::getOldDiag() const{
	if (!_diag_) {
		throw UMAException("The diag_ pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_diag_;
}

const bool &AttrSensor::getIsOriginPure() const{
	return _isOriginPure;
}

const bool &AttrSensor::getObserve() const{
	if (!_observe) {
		throw UMAException("The observe pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_observe;
}

const bool &AttrSensor::getOldObserve() const{
	if (!_observe_) {
		throw UMAException("The observe_ pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_observe_;
}

const bool &AttrSensor::getCurrent() const{
	if (!_current) {
		throw UMAException("The current pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	return *_current;
}

const int &AttrSensor::getIdx() const{
	return _idx;
}

void AttrSensor::setDiag(const double &diag) {
	if (!_diag) {
		throw UMAException("The diag pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	*_diag = diag;
}

void AttrSensor::setOldDiag(const double &diag_) {
	if (!_diag_) {
		throw UMAException("The diag_ pointer is not initiated!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
	}
	*_diag_ = diag_;
}

void AttrSensor::setIsOriginPure(const bool &isOriginPure) {
	_isOriginPure = isOriginPure;
}

AttrSensor::~AttrSensor(){
	pointers_to_null();
}