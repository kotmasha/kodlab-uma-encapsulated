#ifndef _MEASURABLE_
#define _MEASURABLE_

#include "Global.h"
#include "UMACoreObject.h"
class Snapshot;
class Sensor;
class AttrSensorPair;

using namespace std;
/*
This is the AttrSensor class, it is controlled by the Sensor class
*/
class DLL_PUBLIC AttrSensor: public UMACoreObject {
protected:
	//_idx is the index of the measurable in the matrix
	int _idx;
	//_diag is the pointer pointing to the diagonal value of the sensor(weights[i][i])
	double *_diag;
	//_diag_ is the pointer pointing to _diag in last iteration
	double *_diag_;
	//_vdiag is the value of the pointer _diag
	double _vdiag;
	//_vdiag_ is the value of the pointer _diag_
	double _vdiag_;
	//observe pointer, and the value
	bool *_observe, _vobserve;
	//old observe pointer, and the value
	bool*_observe_, _vobserve_;
	//the current pointer
	bool *_current;
	//the target pointer, and the value
	bool *_target, _vtarget;
	//the prediction pointer
	bool *_prediction;
	//indicate if the measurable is originally a pure
	bool _isOriginPure;
	friend class Snapshot;
	friend class SnapshotQualitative;
	friend class Sensor;
	friend class AttrSensorPair;

	friend class GenerateDelayedWeightsTestFixture;
	friend class AmperAndSignalsTestFixture;
	friend class UMACoreDataFlowTestFixture;
	friend class SensorValuePointerConvertionTestFixture;
	friend class AttrSensorSavingLoading;
	friend class UMASavingLoading;

public:
	//AttrSensor(ifstream &file);
	AttrSensor(const string &uuid, UMACoreObject *parent, int idx, bool isOriginPure, double diag);

	void pointersToNull();
	void pointersToValues();
	void valuesToPointers();

	void setDiagPointers(double *_diags, double *_diags_);
	void setObservePointers(bool *observe, bool *observe_);
	void setCurrentPointers(bool *current, bool *current_);
	void setTargetPointers(bool *target);
	void setPredictionPointers(bool *prediction);
	void setIdx(int idx);
	void setObserve(bool observe);
	void setOldObserve(bool observe_);
	
	void saveAttrSensor(ofstream &file);
	static AttrSensor *loadAttrSensor(ifstream &file, UMACoreObject *parent);
	void mergeAttrSensor(AttrSensor * const attrSensor);

	const double &getDiag();
	const double &getOldDiag();
	const bool &getIsOriginPure();
	const bool &getObserve();
	const bool &getOldObserve();
	const bool &getCurrent();
	const int &getIdx() const;
	const bool &getTarget();

	void setDiag(const double &diag);
	void setOldDiag(const double &diag_);
	void setIsOriginPure(const bool &isOriginPure);
	~AttrSensor();
};

#endif