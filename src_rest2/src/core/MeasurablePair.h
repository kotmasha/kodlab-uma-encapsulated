#ifndef _MEASURABLEPAIR_
#define _MEASURABLEPAIR_
#include "Global.h"

class Snapshot;
class SensorPair;
class Measurable;

using namespace std;
/*
This is measurable pair class, owned by sensor pair, manage and store all weight and dir pointers to actual location
variable start with v are temparay variable to store the value, used in amper/delay/pruning and saving/loading
*/
class MeasurablePair{
private:
	//const pointer to measurable, larger idx
	Measurable * _measurable_i;
	//const pointer to measurable, smaller idx
	Measurable * _measurable_j;
	//weight matrix pointer
	double *_w;
	//tmp variable for weight
	double v_w;
	//dir matrix pointer
	bool *_d;
	//tmp variable for dir
	double v_d;
	//friend
	friend class SensorPair;
	friend class Snapshot;

public:
	MeasurablePair(ifstream &file, Measurable *_m_i, Measurable *_m_j);
	MeasurablePair(Measurable *_m_i, Measurable *_m_j, double w, bool d);
	void pointers_to_null();
	void pointers_to_values();
	void values_to_pointers();
	void setWeightPointers(double *weights);
	void setDirPointers(bool *dirs);
	void save_measurable_pair(ofstream &file);
	void copy_data(MeasurablePair *mp);

	double getW();
	bool getD();
	void setW(double &w);
	void setD(bool &d);
	~MeasurablePair();
};

#endif