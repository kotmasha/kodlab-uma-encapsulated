#ifndef _ATTRSENSORPAIR_
#define _ATTRSENSORPAIR_
#include "Global.h"

class Snapshot;
class SensorPair;
class AttrSensor;

using namespace std;
/*
This is measurable pair class, owned by sensor pair, manage and store all weight and dir pointers to actual location
variable start with v are temparay variable to store the value, used in amper/delay/pruning and saving/loading
*/
class DLL_PUBLIC AttrSensorPair{
private:
	//const pointer to measurable, larger idx
	AttrSensor * const _attrSensorI;
	//const pointer to measurable, smaller idx
	AttrSensor * const _attrSensorJ;
	//weight matrix pointer
	double *_w;
	//tmp variable for weight
	double _vw;
	//dir matrix pointer
	bool *_d;
	//tmp variable for dir
	double _vd;
	//friend
	friend class SensorPair;
	friend class Snapshot;
	friend class SnapshotQualitative;

	friend class AmperAndTestFixture;
	friend class GenerateDelayedWeightsTestFixture;
	friend class AmperTestFixture;
	friend class UMACoreDataFlowTestFixture;

public:
	//AttrSensorPair(ifstream &file, AttrSensor *_m_i, AttrSensor *_m_j);
	AttrSensorPair(AttrSensor * const _attrSensorI, AttrSensor * const _attrSensorJ, double w, bool d);

	void pointersToNull();
	void pointersToValues();
	void valuesToPointers();

	void setWeightPointers(double *weights);
	void setDirPointers(bool *dirs);
	//void save_measurable_pair(ofstream &file);
	//void copy_data(AttrSensorPair *mp);

	const double &getW() const;
	const bool &getD() const;
	void setW(const double w);
	void setD(const bool d);
	~AttrSensorPair();
};

#endif