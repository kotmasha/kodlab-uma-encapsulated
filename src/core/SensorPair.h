#ifndef _SENSORPAIR_
#define _SENSORPAIR_

#include "Global.h"

class AttrSensorPair;
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
	AttrSensorPair *mij, *mi_j, *m_ij, *m_i_j;
	//pointer to threshold matrix
	double *_threshold;
	//threshold matrix value
	double _vthreshold;
	
	//Snapshot class should be able to access all SensorPair value
	friend class Snapshot;
	friend class SnapshotQualitative;

	friend class UMACoreDataFlowTestFixture;

public:
	// SensorPair(ifstream &file, vector<Sensor *> &sensors);
	SensorPair(Sensor* const _sensor_i, Sensor* const _sensor_j, double threshold, double total);
	SensorPair(Sensor* const _sensor_i, Sensor* const _sensor_j, double threshold, const vector<double> &w, const vector<bool> &b);
	//init functions
	void setWeightPointers(double *weights);
	void setDirPointers(bool *dirs);
	void setThresholdPointers(double *thresholds);

	void pointersToNull();
	void pointersToValues();
	void valuesToPointers();
	void setAllPointers(double *weights, bool *dirs, double *thresholds);

	void setThreshold(const double &threshold);

	AttrSensorPair *getAttrSensorPair(bool isOriginPure_i, bool isOriginPure_j);
	//void save_sensor_pair(ofstream &file);
	//void copy_data(SensorPair *sp);
	const double &getThreshold() const;
	~SensorPair();
};

#endif