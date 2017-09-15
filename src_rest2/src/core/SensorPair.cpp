#include "SensorPair.h"
#include "MeasurablePair.h"
#include "Sensor.h"

extern int ind(int row, int col);
extern int compi(int x);

SensorPair::SensorPair(ifstream &file, vector<Sensor *> &sensors) {
	int idx_i = -1, idx_j = -1;
	file.read((char *)(&idx_i), sizeof(int));
	file.read((char *)(&idx_j), sizeof(int));
	file.read((char *)(&vthreshold), sizeof(double));
	_sensor_i = sensors[idx_i];
	_sensor_j = sensors[idx_j];
	mij = new MeasurablePair(file, _sensor_i->_m, _sensor_j->_m);
	mi_j = new MeasurablePair(file, _sensor_i->_m, _sensor_j->_cm);
	m_ij = new MeasurablePair(file, _sensor_i->_cm, _sensor_j->_m);
	m_i_j = new MeasurablePair(file, _sensor_i->_cm, _sensor_j->_cm);
	pointers_to_null();
}

/*
init function use sensor pointer, measurable pointer to create measurable pairs
*/
SensorPair::SensorPair(Sensor *_sensor_i, Sensor *_sensor_j, double threshold, double total):
	_sensor_i(_sensor_i),_sensor_j(_sensor_j){
	mij = new MeasurablePair(_sensor_i->_m, _sensor_j->_m, total / 4.0, _sensor_i == _sensor_j);
	mi_j = new MeasurablePair(_sensor_i->_m, _sensor_j->_cm, total / 4.0, false);
	m_ij = new MeasurablePair(_sensor_i->_cm, _sensor_j->_m, total / 4.0, false);
	m_i_j = new MeasurablePair(_sensor_i->_cm, _sensor_j->_cm, total / 4.0, _sensor_i == _sensor_j);
	pointers_to_null();
	this->vthreshold = threshold;
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
	this->threshold = thresholds + ind(_idx_i, _idx_j);
}

/*
This function is validating the SensorPair value from the pointer, used before deleting the h_matrix to retain all necessary data
*/
void SensorPair::pointers_to_values(){	
	vthreshold = *threshold;
	mij->pointers_to_values();
	mi_j->pointers_to_values();
	m_ij->pointers_to_values();
	m_i_j->pointers_to_values();
}

/*
This function is copying the value back to pointer, used when copying sensor value back to GPU
*/
void SensorPair::values_to_pointers(){
	*threshold = vthreshold;
	mij->values_to_pointers();
	mi_j->values_to_pointers();
	m_ij->values_to_pointers();
	m_i_j->values_to_pointers();
}

void SensorPair::pointers_to_null(){
	threshold = NULL;
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
MeasurablePair *SensorPair::getMeasurablePair(bool isOriginPure_i, bool isOriginPure_j){
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
void SensorPair::save_sensor_pair(ofstream &file){
	file.write(reinterpret_cast<const char *>(&_sensor_i->_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_sensor_j->_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&vthreshold), sizeof(double));
	mij->save_measurable_pair(file);
	mi_j->save_measurable_pair(file);
	m_ij->save_measurable_pair(file);
	m_i_j->save_measurable_pair(file);
}

void SensorPair::copy_data(SensorPair *sp) {
	vthreshold = sp->vthreshold;
	mij->copy_data(sp->mij);
	mi_j->copy_data(sp->mi_j);
	m_ij->copy_data(sp->m_ij);
	m_i_j->copy_data(sp->m_i_j);
}

double SensorPair::getThreshold() {
	return vthreshold;
}

/*
destruct sensorpair, sensor destruction is a different process
*/
SensorPair::~SensorPair(){
	pointers_to_null();
	delete mij;
	delete mi_j;
	delete m_ij;
	delete m_i_j;
	mij = NULL;
	mi_j = NULL;
	m_ij = NULL;
	m_i_j = NULL;
}