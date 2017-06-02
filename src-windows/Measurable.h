#ifndef _MEASURABLE_
#define _MEASURABLE_

#include "Global.h"
class Snapshot;
class Sensor;
class MeasurablePair;

using namespace std;
/*
This is the Measurable class, it is controlled by the Sensor class
*/
class Measurable{
protected:
	//the id of the measurable,get from python side
	//char const *_mid;
	//_idx is the index of the measurable in the matrix
	int _idx;
	//_diag is the pointer pointing to the diagonal value of the sensor(weights[i][i])
	//_diag_ is the pointer pointing to _diag in last iteration
	double *_diag, *_diag_;
	//_vdiag is the value of the pointer _diag
	//_vdiag_ is the value of the pointer _diag_
	double _vdiag, _vdiag_;
	//_status is the status of the measurable, turn on or not
	bool *_status;
	//indicate if the measurable is originally a pure
	bool _isOriginPure;
	friend class Snapshot;
	friend class Sensor;
	friend class MeasurablePair;

public:
	Measurable(int idx, bool isOriginPure);
	void pointers_to_null();
	void pointers_to_values();
	void values_to_pointers();
	void setDiagPointers(double *_diags, double *_diags_);
	void setStatusPointers(bool *status);
	void setIdx(int idx);
	void save_measurable(ofstream &file);
	void load_measurable(ifstream &file);
	~Measurable();
};

#endif