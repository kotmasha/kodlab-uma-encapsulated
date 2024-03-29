#ifndef _ATTRSENSORPAIR_
#define _ATTRSENSORPAIR_

#include "Global.h"
#include "UMACoreObject.h"

class Snapshot;
class SensorPair;
class AttrSensor;

using namespace std;
/*
This is measurable pair class, owned by sensor pair, manage and store all weight and dir pointers to actual location
variable start with v are temparay variable to store the value, used in amper/delay/pruning and saving/loading
*/
class DLL_PUBLIC AttrSensorPair: public UMACoreObject{
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
	friend class AttrSensorPairSavingLoading;
	friend class UMASavingLoading;
	friend class UMAAgentCopying;

public:
	AttrSensorPair(UMACoreObject *parent, AttrSensor * const attrSensorI, AttrSensor * const attrSensorJ, double w, bool d);
	AttrSensorPair(const AttrSensorPair &asp, UMACoreObject *parent, AttrSensor * const attrSensorI, AttrSensor * const attrSensorJ);

	void pointersToNull();
	void pointersToValues();
	void valuesToPointers();

	void setWeightPointers(double *weights);
	void setDirPointers(bool *dirs);
	void saveAttrSensorPair(ofstream &file);
	static AttrSensorPair *loadAttrSensorPair(ifstream &file, AttrSensor *attrSensorI, AttrSensor *attrSensorJ,
		bool b, UMACoreObject *parent);
	void mergeAttrSensorPair(AttrSensorPair * const attrSensorPair);

	const double &getW();
	const bool &getD();
	void setW(const double w);
	void setD(const bool d);
	~AttrSensorPair();

protected:
	const string generateUUID(AttrSensor * const _attrSensorI, AttrSensor * const _attrSensorJ) const;
};

#endif