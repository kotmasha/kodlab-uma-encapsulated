#ifndef _MEASURABLE_
#define _MEASURABLE_

#include "Global.h"
class Snapshot;
class Sensor;
class AttrSensorPair;

using namespace std;
/*
This is the AttrSensor class, it is controlled by the Sensor class
*/
class DLL_PUBLIC AttrSensor{
protected:
	//measurable id
	const string _uuid;
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
	//the target pointer
	bool *_target;
	//the prediction pointer
	bool *_prediction;
	//indicate if the measurable is originally a pure
	bool _isOriginPure;
	friend class Snapshot;
	friend class Sensor;
	friend class AttrSensorPair;

	friend class GenerateDelayedWeightsTestFixture;
	friend class AmperAndSignalsTestFixture;
	friend class UMACoreDataFlowTestFixture;

public:
	//AttrSensor(ifstream &file);
	AttrSensor(const string &uuid, int idx, bool isOriginPure, double diag);

	void pointers_to_null();
	void pointers_to_values();
	void values_to_pointers();

	void setDiagPointers(double *_diags, double *_diags_);
	void setObservePointers(bool *observe, bool *observe_);
	void setCurrentPointers(bool *current);
	void setTargetPointers(bool *target);
	void setPredictionPointers(bool *prediction);
	void setIdx(int idx);
	
	//void save_measurable(ofstream &file);
	//void copy_data(AttrSensor *m);

	const double &getDiag() const;
	const double &getOldDiag() const;
	const bool &getIsOriginPure() const;
	const bool &getObserve() const;
	const bool &getOldObserve() const;
	const bool &getCurrent() const;
	const int &getIdx() const;

	void setDiag(const double &diag);
	void setOldDiag(const double &diag_);
	void setIsOriginPure(const bool &isOriginPure);
	~AttrSensor();
};

#endif