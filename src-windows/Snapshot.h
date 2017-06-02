#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include "Global.h"
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

class Measurable;
class Sensor;
class MeasurablePair;
class SensorPair;
/*
----------------Snapshot Base Class-------------------
*/

/*
This is Snapshot class, it is owed by Agent class.
Snapshot is a base class, based on needs, virtual function can have different implementation
The majority of GPU operation start on the Snapshot object, but this object SHOULD NOT be accessed directly by python code directly
All means of operation on Snapshot should go through Agent class
*/
class Snapshot{
protected:
	/*
	-----------------variables used in kernel.cu--------------------------
	variables start with 'host_' means it is used for GPU data copy, but it is on host memory
	variables start with 'dev_' means it is used for GPU computation, it is on device memory
	*/
	bool *h_dirs, *dev_dirs;
	//dir matrix, every pair of measurable has a dir value
	double *h_weights, *dev_weights;
	//weight matrix, every pair of measurable has a weight value
	double *h_thresholds, *dev_thresholds;
	//threshold matrix, every pair of sensor has a threshold
	bool *h_mask_amper, *dev_mask_amper;//amper value collection for mask
	//mask_amper matrix, every sensor has a mask amper, but the construction is all other measurables

	bool *h_observe, *dev_observe;
	//observe array, dealing with the observation from python
	bool *dev_observe_;
	//observe array, storing the observation from python of last iteration
	bool *h_current, *dev_current;
	//current array, storing the observation value after going through propagation
	bool *h_signal, *dev_signal;
	//signal array, hold the input of propagation
	bool *h_load, *dev_load;
	//load array, hold the input of propagation
	bool *h_mask, *dev_mask;
	//mask array, will be calculated during init_mask, which will be used in halucinate
	bool *h_target, *dev_target;
	//target array, used to hold the target result from calculate_target
	double *h_diag, *dev_diag;
	//diagonal value in weight matrix
	double *h_diag_, *dev_diag_;
	//diagonal value in weight matrix of last iteration

	bool *h_prediction;
	//prediction array after the halucinate, have no corresponding device value
	bool *h_up;
	//up array used for separate propagation, have no corresponding device value
	bool *h_down;
	//down array used for separate propagationm have no corresponding device value
	bool *dev_d1, *dev_d2;
	//input used for distance function, no host value
	/*
	-----------------variables used in kernel.cu--------------------------
	*/

protected:
	/*
	variables used in Snapshot class
	all '_max' value depend on the memory_expansion rate
	*/
	const int _type;
	//current Snapshot type
	const int _base_sensor_size;
	//the base sensor size, base sensor is the basic sensor without amper/delay, the value is initiated 
	int _sensor_size;
	//the current sensor size, this value is changable during the test due to amper/delay/pruning
	int _sensor_size_max;
	//the current sensor max size
	int _measurable_size;
	//the measurable size, also changable
	int _measurable_size_max;
	//the measurable max size
	int _sensor2d_size;
	//the sensor 2 dimensional size, hold the size of the threshold
	int _sensor2d_size_max;
	//the _sensor2d_size max value
	int _measurable2d_size;
	//the measurable 2 dimensional size, hold the size of weight and dir matrix
	int _measurable2d_size_max;
	//the _measurable2d_size max value
	int _mask_amper_size;
	//the mask amper size, used to hold the mask amper
	int _mask_amper_size_max;
	//the _mask_amper_size max value

	double _threshold;
	//threshold value from python
	double _phi;
	//phi value from python
	double _q;
	//q value from python
	double _memory_expansion;
	//memory expansion rate, define how large the memory should grow each time when the old memory is not enough to hold all sensors
	string _name;
	//snapshot name
	vector<Sensor*> _sensors;
	//all sensor object
	vector<SensorPair*> _sensor_pairs;
	//all sensor pair object
	int _sensor_num;
	//record current sensor num in the current Snapshot
	int t;
	bool cal_target;
public:
	enum Snapshot_type{STATIONARY, FORGETFUL, UNITTEST};
	double _total, _total_;
	//the total sum(a square block sum, wij+wi_j+w_ij+w_i_j), and the value in last iteration

protected:
	vector<int> convert_list(vector<bool> &list);
	vector<bool> convert_signal_to_sensor(vector<bool> &signal);

public:
	Snapshot(int type, int base_sensor_size, double threshold, string name, vector<string> sensor_names, bool cal_target);
	
	virtual float decide(vector<bool> signal, double phi, bool active);

	void init_size(int sensor_size);
	void init_sensors(vector<string> uuid, vector<string> names);
	void init_sensor_pairs();
	void init_data(vector<string> sensor_names);

	void free_all_parameters();
	void init_pointers();

	void update_state_GPU(bool active);

	virtual void update_weights(bool active);
	virtual void orient_all();
	virtual void update_thresholds();
	virtual void propagate_GPU(bool *signal, bool *load);
	virtual void calculate_total(bool active);
	virtual void calculate_target();
	virtual float distance(bool *d1, bool *d2);

	void up_GPU(vector<bool> signal, bool is_stable);
	void halucinate_GPU();
	void gen_mask();

	/*
	---------------------GET FUNCTION----------------------
	*/
	vector<bool> getCurrent();
	vector<bool> getPrediction();
	vector<bool> getTarget();
	vector<vector<double> > getWeight2D();
	vector<vector<bool> > getDir2D();
	vector<vector<double> > getThreshold2D();
	vector<vector<bool> > getMask_amper2D();
	vector<bool> getMask_amper();
	vector<double> getWeight();
	vector<bool> getDir();
	vector<double> getThreshold();
	vector<bool> getObserve();
	vector<bool> getObserveOld();
	vector<double> getDiag();
	vector<double> getDiagOld();
	vector<bool> getMask();
	vector<bool> getUp();
	vector<bool> getDown();
	Measurable *getMeasurable(int idx);
	MeasurablePair *getMeasurablePair(int m_idx1, int m_idx2);
	string getName();
	/*
	---------------------GET FUNCTION----------------------
	*/

	/*
	---------------------SET FUNCTION----------------------
	*/
	void setTarget(vector<bool> signal);
	void setObserve(vector<bool> observe);
	/*
	---------------------SET FUNCTION----------------------
	*/

	/*
	---------------------COPY FUNCTION---------------------
	before using the copy function, have to make sure all necessary variable are in place
	*/
	void copy_sensor_pair(int start_idx, int end_idx);
	void copy_sensor(int start_idx, int end_idx);
	void copy_mask(vector<bool> mask);
	/*
	---------------------COPY FUNCTION---------------------
	*/

	void generate_delayed_weights(int mid, bool merge);
	void ampers(vector<vector<bool> > lists);
	void amper(vector<int> &list);
	void delays(vector<vector<bool> > list);
	void amperand(int mid1, int mid2, bool merge);
	void pruning(vector<bool> signal);

	void init_direction();
	void init_weight();
	void init_thresholds();
	void init_mask_amper();
	void init_other_parameter();
	void reallocate_memory(int sensor_size);

	void gen_direction();
	void gen_weight();
	void gen_thresholds();
	void gen_mask_amper();
	
	void gen_other_parameters();

	~Snapshot();
};

/*
Stationary Snapshot is not throwing away information when it is not active
*/
class Snapshot_Stationary: public Snapshot{
public:
	Snapshot_Stationary(int base_sensor_size, double threshold, string name, vector<string> sensor_names, double q, bool cal_target);
	virtual ~Snapshot_Stationary();
	virtual void update_weights(bool active);
	virtual void update_thresholds();
	virtual void orient_all();
	virtual void calculate_total(bool active);
	virtual void calculate_target();

protected:
	//double q;
};

/*
Forgetful Snapshot will throw away information when it is not active
*/
class Snapshot_Forgetful: public Snapshot{
public:
	Snapshot_Forgetful(int base_sensor_size, double threshold, string name, vector<string> sensor_names, double q, bool cal_target);
	virtual ~Snapshot_Forgetful();
	virtual void update_weights(bool active);
	virtual void update_thresholds();
	virtual void orient_all();
	virtual void calculate_total(bool active);
	virtual void calculate_target();

protected:
	//double q;
};

/*
UnitTest Snapshot are just used to do unit test, sensor names, snapshot name will not matter here
*/
class Snapshot_UnitTest: public Snapshot{
public:
	Snapshot_UnitTest(int base_sensor_size, double threshold, double q);
	virtual ~Snapshot_UnitTest();
	virtual void update_weights(bool active);
	virtual void orient_all();
	virtual void calculate_total(bool active);
	virtual void calculate_target();

};
#endif
