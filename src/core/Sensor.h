#ifndef _SENSOR_
#define _SENSOR_

#include "Global.h"
#include "UMACoreObject.h"
class Snapshot;
class AttrSensor;
class SensorPair;

using namespace std;
/*
This is the sensor class, it will contain the POINTER to all basic info like measurable, threshold in order to reduce redundancy
Every sensor is distinguished by the sensor id(sid)
//TBD, in distributed environment
*/
class DLL_PUBLIC Sensor: public UMACoreObject {
public:
	//constructors
	Sensor(const string &uuid, UMACoreObject *parent, int idx);
	Sensor(const std::pair<string, string> &idPair, UMACoreObject *parent, const double &total, int idx);
	Sensor(const std::pair<string, string> &idPair, UMACoreObject *parent, const vector<double> &diag, int idx);

	//values and pointer operation
	void valuesToPointers();
	void pointersToValues();
	void pointersToNull();

	//set pointers
	void setAttrSensorDiagPointers(double *diags, double *diags_);
	void setAttrSensorObservePointers(bool *observe, bool *observe_);
	void setAttrSensorCurrentPointers(bool *current, bool *current_);
	void setAttrSensorTargetPointers(bool *target);
	void setAttrSensorPredictionPointers(bool *prediction);

	void setAmperList(int idx);
	void setAmperList(Sensor * const sensor);

	const vector<int> &getAmperList() const;
	bool getObserve() const;
	bool getOldObserve() const;

	void setIdx(int idx);
	const int &getIdx() const;

	void copyAmperList(bool *ampers) const;
	void saveSensor(ofstream &file);
	static Sensor *loadSensor(ifstream &file, UMACoreObject *parent);
	void mergeSensor(Sensor * const sensor);

	bool generateDelayedSignal();

	void setObserveList(bool *observe, bool *observe_);

	~Sensor();

protected:
	//idx is the index of sensor in the array structure, can be changed due to pruning
	int _idx;
	//*m is the pointer to the measurable object under this sensor, and cm is the compi
	AttrSensor *_m, *_cm;
	vector<int> _amper;

	bool *_observe, *_observe_;

	friend class SensorPair;
	friend class Snapshot;
	friend class SnapshotQualitative;
	friend class Simulation;

	friend class UMACoreDataFlowTestFixture;
	friend class SensorValuePointerConvertionTestFixture;
	friend class SensorSavingLoading;
	friend class UMASavingLoading;
	//class SensorPair/Snapshot should be able to access every info of the Sensor
};

#endif