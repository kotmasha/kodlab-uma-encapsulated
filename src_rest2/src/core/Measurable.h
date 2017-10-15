#ifndef _MEASURABLE_
#define _MEASURABLE_

#include "Global.h"
class Snapshot;
class Sensor;
class MeasurablePair;
class Simulation;

using namespace std;
/*
This is the Measurable class, it is controlled by the Sensor class
*/
class Measurable{
protected:
	//the id of the measurable,get from python side
	//char const *_mid;
	string _uuid;
	//_idx is the index of the measurable in the matrix
	int _idx;
	//_diag is the pointer pointing to the diagonal value of the sensor(weights[i][i])
	//_diag_ is the pointer pointing to _diag in last iteration
	double *_diag, *_diag_;
	//_vdiag is the value of the pointer _diag
	//_vdiag_ is the value of the pointer _diag_
	double _vdiag, _vdiag_;
	//_status is the status of the measurable, turn on or not
	bool *_observe, *_observe_;
	bool _vobserve, _vobserve_;
	bool *_current;
	//indicate if the measurable is originally a pure
	bool _isOriginPure;
	friend class Snapshot;
	friend class Sensor;
	friend class MeasurablePair;
	friend class Simulation;

public:
	Measurable(ifstream &file);
	Measurable(string uuid, int idx, bool isOriginPure, double diag);
	void pointers_to_null();
	void pointers_to_values();
	void values_to_pointers();
	void setDiagPointers(double *_diags, double *_diags_);
	void setObservePointers(bool *observe, bool *observe_);
	void setCurrentPointers(bool *current);
	void setIdx(int idx);
	void save_measurable(ofstream &file);
	void copy_data(Measurable *m);

	double getDiag();
	double getOldDiag();
	bool getIsOriginPure();
	bool getObserve();
	bool getOldObserve();
	bool getCurrent();

	void setDiag(double &diag);
	void setOldDiag(double &diag_);
	void setIsOriginPure(bool &isOriginPure);
	~Measurable();
};

#endif