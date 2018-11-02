#include "SensorPair.h"
#include "AttrSensorPair.h"
#include "AttrSensor.h"
#include "Sensor.h"
#include "UMAException.h"
#include "Logger.h"

extern int ind(int row, int col);
extern int compi(int x);

static Logger sensorPairLogger("SensorPair", "log/sensor.log");

//TODO remove it and replace with addAttrSensorPair
SensorPair::SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j) :
	UMACoreObject(generateUUID(sensor_i, sensor_j), UMA_OBJECT::SENSOR_PAIR, parent),
	_sensor_i(sensor_i), _sensor_j(sensor_j) {
}

/*
init function use sensor pointer, attr_sensor pointer to create measurable pairs
*/
SensorPair::SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j, double threshold) :
	UMACoreObject(generateUUID(sensor_i, sensor_j), UMA_OBJECT::SENSOR_PAIR, parent),
	_sensor_i(sensor_i), _sensor_j(sensor_j) {
	mij = new AttrSensorPair(this, _sensor_i->_m, _sensor_j->_m, -1, _sensor_i == _sensor_j);
	mi_j = new AttrSensorPair(this, _sensor_i->_m, _sensor_j->_cm, -1, false);
	m_ij = new AttrSensorPair(this, _sensor_i->_cm, _sensor_j->_m, -1, false);
	m_i_j = new AttrSensorPair(this, _sensor_i->_cm, _sensor_j->_cm, -1, _sensor_i == _sensor_j);
	pointersToNull();
	this->_vthreshold = threshold;

	sensorPairLogger.info("Sensor pair is constructed with total, sid1=" + _sensor_i->_uuid + ", sid2=" + _sensor_j->_uuid, this->getParentChain());
}

/*
init function use sensor pointer, attr_sensor pointer to create measurable pairs
*/
SensorPair::SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j, double threshold, double total):
	UMACoreObject(generateUUID(sensor_i, sensor_j), UMA_OBJECT::SENSOR_PAIR, parent),
	_sensor_i(sensor_i),_sensor_j(sensor_j){
	mij = new AttrSensorPair(this, _sensor_i->_m, _sensor_j->_m, total / 4.0, _sensor_i == _sensor_j);
	mi_j = new AttrSensorPair(this, _sensor_i->_m, _sensor_j->_cm, total / 4.0, false);
	m_ij = new AttrSensorPair(this, _sensor_i->_cm, _sensor_j->_m, total / 4.0, false);
	m_i_j = new AttrSensorPair(this, _sensor_i->_cm, _sensor_j->_cm, total / 4.0, _sensor_i == _sensor_j);
	pointersToNull();
	this->_vthreshold = threshold;

	sensorPairLogger.info("Sensor pair is constructed with total, sid1=" + _sensor_i->_uuid + ", sid2=" + _sensor_j->_uuid, this->getParentChain());
}

SensorPair::SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j, double threshold, const vector<double> &w, const vector<bool> &b):
	UMACoreObject(generateUUID(sensor_i, sensor_j), UMA_OBJECT::SENSOR_PAIR, parent),
	_sensor_i(sensor_i), _sensor_j(sensor_j) {
	mij = new AttrSensorPair(this, _sensor_i->_m, _sensor_j->_m, w[0], b[0]);
	mi_j = new AttrSensorPair(this, _sensor_i->_m, _sensor_j->_cm, w[1], b[1]);
	m_ij = new AttrSensorPair(this, _sensor_i->_cm, _sensor_j->_m, w[2], b[2]);
	m_i_j = new AttrSensorPair(this, _sensor_i->_cm, _sensor_j->_cm, w[3], b[3]);
	pointersToNull();
	this->_vthreshold = threshold;

	sensorPairLogger.info("Sensor pair is constructed with weights and dirs, sid1=" + _sensor_i->_uuid + ", sid2=" + _sensor_j->_uuid, this->getParentChain());
}

/*
This function is setting the weight pointers
*/
void SensorPair::setWeightPointers(double *weights){
	mij->setWeightPointers(weights);
	mi_j->setWeightPointers(weights);
	m_ij->setWeightPointers(weights);
	m_i_j->setWeightPointers(weights);
}

/*
This function is setting the dir pointers
*/
void SensorPair::setDirPointers(bool *dirs){
	mij->setDirPointers(dirs);
	mi_j->setDirPointers(dirs);
	m_ij->setDirPointers(dirs);
	m_i_j->setDirPointers(dirs);
}

/*
This function is setting the threshold pointers
*/
void SensorPair::setThresholdPointers(double *thresholds){
	int _idx_i = _sensor_i->_idx;
	int _idx_j = _sensor_j->_idx;
	this->_threshold = thresholds + ind(_idx_i, _idx_j);
}

/*
This function is validating the SensorPair value from the pointer, used before deleting the h_matrix to retain all necessary data
*/
void SensorPair::pointersToValues(){	
	_vthreshold = *_threshold;
	mij->pointersToValues();
	mi_j->pointersToValues();
	m_ij->pointersToValues();
	m_i_j->pointersToValues();
}

/*
This function is copying the value back to pointer, used when copying sensor value back to GPU
*/
void SensorPair::valuesToPointers(){
	*_threshold = _vthreshold;
	mij->valuesToPointers();
	mi_j->valuesToPointers();
	m_ij->valuesToPointers();
	m_i_j->valuesToPointers();
}

void SensorPair::pointersToNull(){
	_threshold = NULL;
	mij->pointersToNull();
	mi_j->pointersToNull();
	m_ij->pointersToNull();
	m_i_j->pointersToNull();
}

/*
This function is doing copy value back to GPU when amper/dealy is done
Input: weight, dir, threshold matrix address on host
*/
void SensorPair::setAllPointers(double *weights, bool *dirs, double *thresholds){
	setWeightPointers(weights);
	setDirPointers(dirs);
	setThresholdPointers(thresholds);
}

/*
This function is getting the measurable pair by its pure/compi
*/
AttrSensorPair *SensorPair::getAttrSensorPair(bool isOriginPure_i, bool isOriginPure_j){
	if(isOriginPure_i && isOriginPure_j) return mij;
	else if(isOriginPure_i && !isOriginPure_j) return mi_j;
	else if(!isOriginPure_i && isOriginPure_j) return m_ij;
	else return m_i_j;
}

/*
This function is save sensor pair
Saving order MUST FOLLOW:
1 first sensor idx
2 second sensor idx
3 threshold value
4 mij measurable pair
5 mi_j measurable pair
6 m_ij measurable pair
7 m_i_j measurable pair
*/
void SensorPair::saveSensorPair(ofstream &file){
	file.write(reinterpret_cast<const char *>(&_sensor_i->_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_sensor_j->_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_vthreshold), sizeof(double));
	mij->saveAttrSensorPair(file);
	mi_j->saveAttrSensorPair(file);
	m_ij->saveAttrSensorPair(file);
	m_i_j->saveAttrSensorPair(file);
}

SensorPair *SensorPair::loadSensorPair(ifstream &file, vector<Sensor*> sensors, UMACoreObject *parent) {
	int idxI = -1, idxJ = -1;
	double vthreshold = 0.0;
	file.read((char *)(&idxI), sizeof(int));
	file.read((char *)(&idxJ), sizeof(int));
	file.read((char *)(&vthreshold), sizeof(double));

	Sensor *sensorI = sensors[idxI];
	Sensor *sensorJ = sensors[idxJ];

	SensorPair *sensorPair = new SensorPair(parent, sensorI, sensorJ);
	sensorPair->_vthreshold = vthreshold;

	sensorPair->mij = AttrSensorPair::loadAttrSensorPair(file, sensorI->_m, sensorJ->_m, true, sensorPair);
	sensorPair->mi_j = AttrSensorPair::loadAttrSensorPair(file, sensorI->_m, sensorJ->_cm, false, sensorPair);
	sensorPair->m_ij = AttrSensorPair::loadAttrSensorPair(file, sensorI->_cm, sensorJ->_m, false, sensorPair);
	sensorPair->m_i_j = AttrSensorPair::loadAttrSensorPair(file, sensorI->_cm, sensorJ->_cm, true, sensorPair);

	sensorPair->pointersToNull();

	return sensorPair;
}

/*
void SensorPair::copy_data(SensorPair *sp) {
	vthreshold = sp->vthreshold;
	mij->copy_data(sp->mij);
	mi_j->copy_data(sp->mi_j);
	m_ij->copy_data(sp->m_ij);
	m_i_j->copy_data(sp->m_i_j);
}
*/

void SensorPair::setThreshold(const double &threshold) {
	if (!_threshold) {
		throw UMABadOperationException("The threshold pointer is not initiated!", false, &sensorPairLogger, this->getParentChain());
	}
	*_threshold = threshold;
}

const double &SensorPair::getThreshold(){
	if (!_threshold) {
		throw UMABadOperationException("The threshold pointer is not initiated!", false, &sensorPairLogger, this->getParentChain());
	}
	return *_threshold;
}

/*
destruct sensorpair, sensor destruction is a different process
*/
SensorPair::~SensorPair(){
	pointersToNull();
	delete mij;
	delete mi_j;
	delete m_ij;
	delete m_i_j;
	mij = NULL;
	mi_j = NULL;
	m_ij = NULL;
	m_i_j = NULL;
}

const string SensorPair::generateUUID(Sensor* const _sensor_i, Sensor* const _sensor_j) const {
	return _sensor_i->_uuid + "-" + _sensor_j->_uuid;
}