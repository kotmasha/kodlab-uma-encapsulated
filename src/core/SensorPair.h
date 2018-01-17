#ifndef _SENSORPAIR_
#define _SENSORPAIR_

#include "Global.h"

class MeasurablePair;
class Sensor;

using namespace std;
/*
SensorPair is functioning as the manager for a pair of sensors
It is distinguished by 2 sensor ids, and it will point to values like weight, dir and thresholds
*/
class DLL_PUBLIC SensorPair{
private:
	//const pointer to sensor, larger idx
	Sensor * _sensor_i;
	//const pointer to sensor, smaller idx
	Sensor * _sensor_j;
	//pointers to measurable pairs
	MeasurablePair *mij, *mi_j, *m_ij, *m_i_j;
	//pointer to threshold matrix
	double *threshold;
	//threshold matrix value
	double vthreshold;
	
	//Snapshot class should be able to access all SensorPair value
	friend class Snapshot;

public:
	// SensorPair(ifstream &file, vector<Sensor *> &sensors);
	SensorPair(Sensor* const _sensor_i, Sensor* const _sensor_j, double threshold, double total);
	SensorPair(Sensor* const _sensor_i, Sensor* const _sensor_j, double threshold, const vector<double> &w, const vector<bool> &b);
	//init functions
	void setWeightPointers(double *weights);
	void setDirPointers(bool *dirs);
	void setThresholdPointers(double *thresholds);

	void pointers_to_null();
	void pointers_to_values();
	void values_to_pointers();
	void setAllPointers(double *weights, bool *dirs, double *thresholds);

	MeasurablePair *getMeasurablePair(bool isOriginPure_i, bool isOriginPure_j);
	//void save_sensor_pair(ofstream &file);
	//void copy_data(SensorPair *sp);
	const double &getThreshold() const;
	~SensorPair();
};

#endif