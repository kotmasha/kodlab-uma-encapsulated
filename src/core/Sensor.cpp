#include "Sensor.h"
#include "AttrSensor.h"
#include "Logger.h"
#include "UMAException.h"

extern int ind(int row, int col);
extern int compi(int x);

static Logger sensorLogger("Sensor", "log/sensor.log");

/*
Init function
Input: _sid is sensor id, const int, and _sname, sensor name
*/
Sensor::Sensor(const std::pair<string, string> &idPair, UMACoreObject *parent, const double &total, int idx)
	:UMACoreObject(idPair.first, UMA_OBJECT::SENSOR, parent){
	_idx = idx;
	_m = new AttrSensor(idPair.first, this, 2 * idx, true, total / 2);
	_cm = new AttrSensor(idPair.second, this, 2 * idx + 1, false, total / 2);
	_observe = nullptr;
	_observe_ = nullptr;

	sensorLogger.info("New sensor created with total value, id=" + _uuid, this->getParentChain());
}

Sensor::Sensor(const std::pair<string, string> &idPair, UMACoreObject *parent, const vector<double> &diag, int idx)
	: UMACoreObject(idPair.first, UMA_OBJECT::SENSOR, parent) {
	_idx = idx;
	_m = new AttrSensor(idPair.first, this, 2 * idx, true, diag[0]);
	_cm = new AttrSensor(idPair.second, this, 2 * idx + 1, false, diag[1]);
	_observe = nullptr;
	_observe_ = nullptr;

	sensorLogger.info("New sensor created with diag value, id=" + _uuid, this->getParentChain());
}

Sensor::Sensor(const string &uuid, UMACoreObject *parent, int idx)
	: UMACoreObject(uuid, UMA_OBJECT::SENSOR, parent) {
	_idx = idx;
	_observe = nullptr;
	_observe_ = nullptr;
}

Sensor::Sensor(const Sensor &sensor, UMACoreObject *parent)
	: UMACoreObject(sensor._uuid, UMA_OBJECT::SENSOR, parent) {
	_idx = sensor._idx;
	_amper = sensor._amper;

	_m = new AttrSensor(*sensor._m, this);
	_cm = new AttrSensor(*sensor._cm, this);
	_observe = nullptr;
	_observe_ = nullptr;

	sensorLogger.debug("A new sensor is copied, id=" + _uuid, this->getParentChain());
}

/*
This function is copying the diag value of _m and _cm to the pointer
*/
void Sensor::valuesToPointers() {
	_m->valuesToPointers();
	_cm->valuesToPointers();
}

/*
This function is copying the pointer of _m and _cm to value
*/
void Sensor::pointersToValues() {
	_m->pointersToValues();
	_cm->pointersToValues();
}

/*
This function is setting pointers to NULL
*/
void Sensor::pointersToNull() {
	_m->pointersToNull();
	_cm->pointersToNull();
}

/*
This function is setting the diag pointers of _m and _cm through the sensor object
Input: weight matrix diagonal value of this and last iteration
*/
void Sensor::setAttrSensorDiagPointers(double *diags, double *diags_) {
	_m->setDiagPointers(diags, diags_);
	_cm->setDiagPointers(diags, diags_);
}

/*
This function is setting the status pointers of _m and _cm through the sensor object
Input: observe value of this and last iteration
*/
void Sensor::setAttrSensorObservePointers(bool *observe, bool *observe_) {
	_m->setObservePointers(observe, observe_);
	_cm->setObservePointers(observe, observe_);
	_observe = observe;
	_observe_ = observe_;
}

/*
This function is setting the current pointers of _m and _cm
Input: current pointer
*/
void Sensor::setAttrSensorCurrentPointers(bool *current, bool *current_) {
	_m->setCurrentPointers(current, current_);
	_cm->setCurrentPointers(current, current_);
}

/*
This function is setting the current pointers of _m and _cm
Input: target pointer
*/
void Sensor::setAttrSensorTargetPointers(bool *target) {
	_m->setTargetPointers(target);
	_cm->setTargetPointers(target);
}

/*
This function is setting the current pointers of _m and _cm
Input: prediction pointer
*/
void Sensor::setAttrSensorPredictionPointers(bool *prediction) {
	_m->setPredictionPointers(prediction);
	_cm->setPredictionPointers(prediction);
}

/*
This is setting the amper list, used when the ampers to be copied is permanant
Input: idx to be added in the amper
*/
void Sensor::setAmperList(int idx){
	_amper.push_back(idx);
}

/*
This is setting the amper list, using a new sensor pointer, just copy what is inside the _amper to the current Sensor
Input: pointer of another sensor
*/
void Sensor::setAmperList(Sensor * const sensor){
	_amper.insert(_amper.end(), sensor->_amper.begin(), sensor->_amper.end());
}


const vector<int> &Sensor::getAmperList() const {
	return _amper;
}

/*
Get the sensor idx
*/
const int &Sensor::getIdx() const {
	return _idx;
}

/*
This function is changing the idx when a pruning happens, the corresponding measurable idx also need to change
Input: the new idx of the sensor
*/
void Sensor::setIdx(int idx) {
	_idx = idx;
	_m->setIdx(2 * _idx);
	_cm->setIdx(2 * _idx + 1);
}

bool Sensor::getObserve() const {
	return _m->getObserve();
}

bool Sensor::getOldObserve() const {
	return _m->getOldObserve();
}

/*
This is copying amper list into the amper array
Input: the ampers array pointer
*/
void Sensor::copyAmperList(bool *ampers) const{
	int offset = 2 * ind(_idx, 0);
	for(int i = 2 * ind(_idx, 0); i < 2 * ind(_idx + 1, 0); ++i){
		//by default, the copy is happening to the row of the sid
		ampers[i] = false;
	}
	for(int i = 0; i < _amper.size(); ++i){
		ampers[_amper[i] + offset] = true;
	}
}

/*
This is the save sensor function
Saving order MUST FOLLOW:
1 sid of the sensor
2 sensor name
3 sensor idx
4 amper list
    4.1 the size of the amper list
	4.2 all amper list value based on the size
5 all measurables
Input: ofstream file
*/
void Sensor::saveSensor(ofstream &file){
	//write sid and idx
	int uuidLength = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuidLength), sizeof(int));
	file.write(_uuid.c_str(), uuidLength * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_idx), sizeof(int));
	//write the amper list
	int amperSize = _amper.size();
	file.write(reinterpret_cast<const char *>(&amperSize), sizeof(int));
	for(int i = 0; i < amperSize; ++i){
		file.write(reinterpret_cast<const char *>(&_amper[i]), sizeof(int));
	}
	_m->saveAttrSensor(file);
	_cm->saveAttrSensor(file);
}

Sensor *Sensor::loadSensor(ifstream &file, UMACoreObject *parent) {
	int uuidLength = -1;
	file.read((char *)(&uuidLength), sizeof(int));

	string uuid = string(uuidLength, ' ');
	file.read(&uuid[0], uuidLength * sizeof(char));

	int idx = -1;
	file.read((char *)(&idx), sizeof(int));

	int amperSize = -1;
	vector<int> amper;
	file.read((char *)(&amperSize), sizeof(int));
	for (int i = 0; i < amperSize; ++i) {
		int tmp = -1;
		file.read((char *)(&tmp), sizeof(int));
		amper.push_back(tmp);
	}

	Sensor *sensor = new Sensor(uuid, parent, idx);
	sensor->_amper = amper;
	sensor->_m = AttrSensor::loadAttrSensor(file, sensor);
	sensor->_cm = AttrSensor::loadAttrSensor(file, sensor);
	
	return sensor;
}

/*
This function is copying the data from the sensor to current sensor
Input: sensor
*/
void Sensor::mergeSensor(Sensor * const sensor){
	_m->mergeAttrSensor(sensor->_m);
	_cm->mergeAttrSensor(sensor->_cm);
}

bool Sensor::generateDelayedSignal() {
	if (!_observe_) {
		throw UMABadOperationException("the old observe signal is NULL, be sure to init the sensor first!", false, &sensorLogger, this->getParentChain());
	}
	for (int i = 0; i < _amper.size(); ++i) {
		int j = _amper[i];
		if (!_observe[j]) return false;
	}
	return true;
}

void Sensor::setObserveList(bool *observe, bool *observe_) {
	_observe = observe;
	_observe_ = observe_;
}

/*
destruct the sensor, sensorpair destruction is a separate process
*/
Sensor::~Sensor(){
	//destruct measurable first
	delete _m;
	delete _cm;
	_m = NULL;
	_cm = NULL;
	_amper.clear();
}