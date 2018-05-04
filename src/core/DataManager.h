#ifndef _DATAMANAGER_
#define _DATAMANAGER_

#include "Global.h"

class Sensor;
class SensorPair;

using namespace std;

class DLL_PUBLIC DataManager {
public:
	enum { WEIGHTS, DIRS, NPDIRS, THRESHOLDS, DISTS, NPDIR_MASK, MASK_AMPER, MASK,
		CURRENT, OLD_CURRENT, OBSERVE, PREDICTION, TARGET, NEGLIGIBLE, SIGNAL,
		SIGNALS, LSIGNALS, LOAD, DIAG, OLD_DIAG, UNION_ROOT, RES, DEC_TMP1, DEC_TMP2,
		SUM, BOOL_TMP };
protected:
	/*
	-----------------variables used in kernel.cu--------------------------
	variables start with 'host_' means it is used for GPU data copy, but it is on host memory
	variables start with 'dev_' means it is used for GPU computation, it is on device memory
	*/
	//dir matrix, every pair of attr_sensor has a dir value
	bool *h_dirs, *dev_dirs;
	//weight matrix, every pair of attr_sensor has a weight value
	double *h_weights, *dev_weights;
	//threshold matrix, every pair of sensor has a threshold
	double *h_thresholds, *dev_thresholds;
	//amper value collection for mask, mask_amper matrix, every sensor has a mask amper, but the construction is all other attr_sensors
	bool *h_mask_amper, *dev_mask_amper;
	//n power of dir matrix, computed using floyd algorithm									
	bool *h_npdirs, *dev_npdirs;
	
	//observe array, dealing with the observation from python
	bool *h_observe, *dev_observe;
	//old observe array, storing the observation from python of last iteration
	bool *h_observe_;
	//current array, storing the observation value after going through propagation
	bool *h_current, *dev_current;
	//old current array, storing the current value of last iteration
	bool *h_current_, *dev_current_;
	//load array, hold the input of propagation
	bool *h_load, *dev_load;
	//mask array, will be calculated during init_mask, which will be used in halucinate
	bool *h_mask, *dev_mask;
	//target array, used to hold the target result from calculate_target
	bool *h_target, *dev_target;
	//the negligible sensor
	bool *h_negligible, *dev_negligible;
	//diagonal value in weight matrix
	double *h_diag, *dev_diag;
	//old diagonal value in weight matrix of last iteration
	double *h_diag_, *dev_diag_;
	//mask propgated on npdir
	bool *h_npdir_mask, *dev_npdir_mask;
	//signals
	bool *h_signals, *dev_signals;
	//loaded signals
	bool *h_lsignals, *dev_lsignals;
	//bool tmp signals
	bool *h_bool_tmp, *dev_bool_tmp;
	//distance for block gpu
	int *h_dists, *dev_dists;
	//union-find root
	int *h_union_root, *dev_union_root;

	//prediction array after the halucinate, have no corresponding device value
	bool *h_prediction, *dev_prediction;
	//decision tmp variables
	bool *dev_dec_tmp1;
	bool *dev_dec_tmp2;
	//result variable for distance/divergence
	float *h_res, *dev_res;
	//a variable for sum function
	double *dev_sum;
	/*
	-----------------variables used in kernel.cu--------------------------
	*/

protected:
	//the current sensor size, this value is changable during the test due to amper/delay/pruning
	int _sensor_size;
	//the current sensor max size
	int _sensor_size_max;
	//the attr_sensor size, also changable
	int _attr_sensor_size;
	//the attr_sensor max size
	int _attr_sensor_size_max;
	//the sensor 2 dimensional size, hold the size of the threshold
	int _sensor2d_size;
	//the _sensor2d_size max value
	int _sensor2d_size_max;
	//the attr_sensor 2 dimensional size, hold the size of weight and dir matrix
	int _attr_sensor2d_size;
	//the _attr_sensor2d_size max value
	int _attr_sensor2d_size_max;
	//the mask amper size, used to hold the mask amper
	int _mask_amper_size;
	//the _mask_amper_size max value
	int _mask_amper_size_max;
	//the npdir matrix size, npdir_size = _attr_sensor2d_size + sensor_size(increase value on y=2i, x=2i)
	int _npdir_size;
	//the npdir max size
	int _npdir_size_max;
	//memory expansion rate, define how large the memory should grow each time when the old memory is not enough to hold all sensors
	double _memory_expansion;
	//the dataManager's dependency chain
	const string _dependency;
	//the cublas handle

public:
	DataManager(const string &dependency);

	int *_dvar_i(int name);
	double *_dvar_d(int name);
	bool *_dvar_b(int name);
	float *_dvar_f(int name);

	int *_hvar_i(int name);
	double *_hvar_d(int name);
	bool *_hvar_b(int name);
	float *_hvar_f(int name);

	//#############
	//Set Functions
	void setMask(const vector<bool> &mask);
	void setLoad(const vector<bool> &load);
	void setSignals(const vector<vector<bool> > &signals);
	void setLSignals(const vector<vector<bool> > &signals);
	void setDists(const vector<vector<int> > &dists);
	void setObserve(const vector<bool> &observe);
	void setCurrent(const vector<bool> &current);
	void setOldCurrent(const vector<bool> &current);
	void setTarget(const vector<bool> &signal);
	//Set Functions
	//#############

	//#############
	//Get Functions
	const vector<bool> getCurrent();
	const vector<bool> getOldCurrent();
	const vector<bool> getPrediction();
	const vector<bool> getTarget();
	const vector<vector<double> > getWeight2D();
	const vector<vector<bool> > getDir2D();
	const vector<vector<bool> > getNPDir2D();
	const vector<vector<double> > getThreshold2D();
	const vector<vector<bool> > getMask_amper2D();
	const vector<bool> getMask_amper();
	const vector<double> getWeight();
	const vector<bool> getDir();
	const vector<bool> getNPDir();
	const vector<double> getThresholds();
	const vector<bool> getObserve();
	const vector<double> getDiag();
	const vector<double> getDiagOld();
	const vector<bool> getMask();
	const vector<bool> getNegligible();
	const vector<bool> getTmpBool();
	const vector < vector<bool> > getSignals(int sig_count);
	const vector < vector<bool> > getLSignals(int sig_count);
	const vector<vector<bool> > getNpdirMasks();
	const vector<bool> getLoad();
	const vector<int> getUnionRoot();
	const std::map<string, int> getSizeInfo();
	//Get Functions
	//#############

	~DataManager();

	friend class Snapshot;

	friend class UMACoreDataFlowTestFixture;

protected:
	void init_pointers();
	void reallocate_memory(double &total, int sensor_size);
	void set_size(int sensor_size, bool change_max=true);
	void free_all_parameters();

	void gen_direction();
	void gen_weight();
	void gen_thresholds();
	void gen_mask_amper();
	void gen_np_direction();
	void gen_signals();
	void gen_npdir_mask();
	void gen_dists();
	void gen_other_parameters();

	void init_other_parameter(double &total);

	void create_sensors_to_arrays_index(const int start_idx, const int end_idx, const vector<Sensor*> &sensors);
	void create_sensor_pairs_to_arrays_index(const int start_idx, const int end_idx, const vector<SensorPair*> &sensors);

	void copy_sensor_pairs_to_arrays(const int start_idx, const int end_idx, const vector<SensorPair*> &_sensor_pairs);
	void copy_sensors_to_arrays(const int start_idx, const int end_idx, const vector<Sensor*> &sensors);
	void copy_arrays_to_sensors(const int start_idx, const int end_idx, const vector<Sensor*> &sensors);
	void copy_arrays_to_sensor_pairs(const int start_idx, const int end_idx, const vector<SensorPair*> &_sensor_pairs);
};

#endif