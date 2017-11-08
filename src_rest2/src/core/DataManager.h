#ifndef _DATAMANAGER_
#define _DATAMANAGER_

#include "Global.h"

class Sensor;
class SensorPair;
class logManager;

using namespace std;

class DataManager {
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
	bool *h_npdirs, *dev_npdirs;
	//n power of dir matrix, computed using floyd algorithm

	bool *h_observe, *dev_observe;
	//observe array, dealing with the observation from python
	bool *h_observe_;
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
	bool *h_negligible, *dev_negligible;
	//the negligible sensor
	double *h_diag, *dev_diag;
	//diagonal value in weight matrix
	double *h_diag_, *dev_diag_;
	//diagonal value in weight matrix of last iteration
	bool *h_npdir_mask, *dev_npdir_mask;
	//mask propgated on npdir
	bool *h_signals, *dev_signals;
	//signals
	bool *h_lsignals, *dev_lsignals;
	//loaded signals
	int *h_dists, *dev_dists;
	//distance for block gpu
	int *h_union_root, *dev_union_root;

	bool *h_prediction, *dev_prediction;
	//prediction array after the halucinate, have no corresponding device value
	bool *h_up;
	//up array used for separate propagation, have no corresponding device value
	bool *h_down;
	//down array used for separate propagationm have no corresponding device value
	/*
	-----------------variables used in kernel.cu--------------------------
	*/

protected:
	/*
	variables used in Snapshot class
	all '_max' value depend on the memory_expansion rate
	*/
	int _type;
	//current Snapshot type
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
	int _npdir_size;
	//the npdir matrix size, npdir_size = _measurable2d_size + sensor_size(increase value on y=2i, x=2i)
	int _npdir_size_max;
	//the npdir max size
	double _memory_expansion;
	//memory expansion rate, define how large the memory should grow each time when the old memory is not enough to hold all sensors
	int _initial_sensor_size;
	//initial sensor size for the whole test
	
protected:
	logManager *_log;
	string _log_dir;

public:
	DataManager(string &log_dir);

	int *_dvar_i(int name);
	double *_dvar_d(int name);
	bool *_dvar_b(int name);

	int *_hvar_i(int name);
	double *_hvar_d(int name);
	bool *_hvar_b(int name);

	/*
	Set Functions
	*/
	void setMask(vector<bool> &mask);
	void setLoad(vector<bool> &load);
	void setSignal(vector<bool> &signal);
	void setSignals(vector<vector<bool> > &signals);
	void setLSignals(vector<vector<bool> > &signals);
	void setDists(vector<vector<int> > &dists);
	void setObserve(vector<bool> &observe);
	void setCurrent(vector<bool> &current);
	void setTarget(vector<bool> &signal);

	vector<bool> getCurrent();
	vector<bool> getPrediction();
	vector<bool> getTarget();
	vector<vector<double> > getWeight2D();
	vector<vector<bool> > getDir2D();
	vector<vector<bool> > getNPDir2D();
	vector<vector<double> > getThreshold2D();
	vector<vector<bool> > getMask_amper2D();
	vector<bool> getMask_amper();
	vector<double> getWeight();
	vector<bool> getDir();
	vector<bool> getNPDir();
	vector<double> getThresholds();
	vector<bool> getObserve();
	vector<double> getDiag();
	vector<double> getDiagOld();
	vector<bool> getMask();
	vector<bool> getUp();
	vector<bool> getDown();
	vector<bool> getNegligible();
	vector < vector<bool> > getSignals(int sig_count);
	vector < vector<bool> > getLSignals(int sig_count);
	vector<vector<bool> > getNpdirMasks();
	vector<bool> getLoad();
	vector<int> getUnionRoot();
	std::map<string, int> getSizeInfo();
	int getSensorSize();
	int getMeasurableSize();
	int getMeasurable2dSize();

	friend class Snapshot;

protected:
	void init_pointers();
	void reallocate_memory(double &total, int sensor_size);
	void set_size(int sensor_size, bool change_max);
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

	void create_sensors_to_arrays_index(int start_idx, int end_idx, vector<Sensor*> &sensors);
	void create_sensor_pairs_to_arrays_index(int start_idx, int end_idx, vector<SensorPair*> &sensors);

	void copy_sensor_pairs_to_arrays(int start_idx, int end_idx, vector<SensorPair*> &_sensor_pairs);
	void copy_sensors_to_arrays(int start_idx, int end_idx, vector<Sensor*> &sensors);
	void copy_arrays_to_sensors(int start_idx, int end_idx, vector<Sensor*> &sensors);
	void copy_arrays_to_sensor_pairs(int start_idx, int end_idx, vector<SensorPair*> &_sensor_pairs);
};

#endif