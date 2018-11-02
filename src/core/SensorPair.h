#ifndef _SENSORPAIR_
#define _SENSORPAIR_

#include "Global.h"
#include "UMACoreObject.h"

class AttrSensorPair;
class Sensor;

using namespace std;
/*
SensorPair is functioning as the manager for a pair of sensors
It is distinguished by 2 sensor ids, and it will point to values like weight, dir and thresholds
*/
class DLL_PUBLIC SensorPair: public UMACoreObject{
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
	friend class SensorPairSavingLoading;

public:
	SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j);
	SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j, double threshold);
	SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j, double threshold, double total);
	SensorPair(UMACoreObject *parent, Sensor* const sensor_i, Sensor* const sensor_j, double threshold, const vector<double> &w, const vector<bool> &b);
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
	void saveSensorPair(ofstream &file);
	static SensorPair *loadSensorPair(ifstream &file, vector<Sensor*> sensors, UMACoreObject *parent);
	//void copy_data(SensorPair *sp);
	const double &getThreshold();
	~SensorPair();

protected:
	const string generateUUID(Sensor* const _sensor_i, Sensor* const _sensor_j) const;
};

#endif