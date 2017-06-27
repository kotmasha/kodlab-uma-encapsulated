#ifndef _SENSOR_
#define _SENSOR_
#include "Global.h"
class Snapshot;
class Measurable;
class SensorPair;
class logManager;

using namespace std;
/*
This is the sensor class, it will contain the POINTER to all basic info like measurable, threshold in order to reduce redundancy
Every sensor is distinguished by the sensor id(sid)
//TBD, in distributed environment
*/
class Sensor{
protected:
	//sid is the id of the sensor, get from python side, not changable
	string _sid;
	//idx is the index of sensor in the array structure, can be changed due to pruning
	int _idx;
	//the given sensor name
	string _sname;
	//*m is the pointer to the measurable object under this sensor, and cm is the compi
	Measurable *_m, *_cm;
	vector<int> _amper;

	friend class SensorPair;
	friend class Snapshot;
	//class SensorPair/Snapshot should be able to access every info of the Sensor
public:
	Sensor(ifstream &file);
	Sensor(string sid, string sname, int idx);

	void setMeasurableDiagPointers(double *_diags, double *_diags_);
	void setMeasurableStatusPointers(bool *current);
	void values_to_pointers();
	void pointers_to_values();
	void init_amper_list(bool *ampers);
	void setAmperList(bool *ampers, int start, int end);
	void setAmperList(int idx);
	void setAmperList(Sensor *sensor);
	void setIdx(int idx);
	void copyAmperList(bool *ampers);
	bool amper_and_signals(bool *observe);
	bool isSensorActive();
	void pointers_to_null();
	void save_sensor(ofstream &file);

	~Sensor();
};

#endif