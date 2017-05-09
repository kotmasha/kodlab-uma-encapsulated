#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;
/*
----------------Agent Base Class-------------------
*/

class logging;
class AgentDM;

class Agent{
protected:
	/*
	-----------------variables used in kernel.cu--------------------------
	*/
	bool *Gdir, *dev_dir;//dir is DIR in python
	double *Gweights, *dev_weights;
	double *Gthresholds, *dev_thresholds;//weight and threshold in python
	bool *GMask_amper, *dev_mask_amper;//amper value collection for mask

	bool *Gobserve, *dev_observe, *Gobserve_old, *dev_observe_old;//observe in python
	bool *Gsignal, *dev_signal, *Gload, *dev_load;//signal and load variable in propagate
	bool *Gcurrent, *dev_current;//current in python
	bool *Gmask, *dev_mask;//bool value for mask signal in halucinate
	bool *Gtarget, *dev_target;//bool value for the target list
	bool *Gprediction;//value after halucinate
	double *GMeasurable, *dev_measurable;//bool value for every measurable value on diagonal
	double *GMeasurable_old, *dev_measurable_old;
	bool *Gup, *Gdown;

	bool *dev_signal1, *dev_signal2;
	/*
	-----------------variables used in kernel.cu--------------------------
	*/

protected:
	//general variables for agent
	int sensor_size, base_sensor_size, sensor_size_max;
	int measurable_size, measurable_size_max;
	int array_size, array_size_max;
	int mask_amper_size, mask_amper_size_max;
	int t;
	double threshold, phi, total, last_total, q;
	bool print_info;
	string name;
	int type;
	double memory_expansion;
	vector<string> sensors_names, decision;
	logging *logging_info;
	friend class AgentDM;
	AgentDM *agentDM;
	//general variables for agent

public:
	enum agent_type{STATIONARY, FORGETFUL};

public:
	Agent(int type, bool print_info);
	virtual ~Agent();

	float decide(vector<bool> signal, double phi, bool active);

	void init_data(string name, int sensorSize, vector<string> sensors_names, string filename);
	void free_all_parameters();
	void init_pointers();

	void update_state_GPU(bool active);

	virtual void update_weights(bool active);
	virtual void orient_all();
	virtual void update_thresholds();

	void propagate_GPU(vector<bool> signal, vector<bool> load, bool t);

	virtual void calculate_total(bool active);
	virtual void calculate_target();

	float distance(bool *signal1, bool *signal2);
	float distance_big(bool *signal1, bool *signal2);

	void up_GPU(vector<bool> signal, bool is_stable);
	void halucinate_GPU();
	void gen_mask();

	/*
	---------------------GET FUNCTION----------------------
	*/
	vector<bool> getCurrent();
	vector<bool> getPrediction();
	vector<bool> getSignal();
	vector<bool> getLoad();
	vector<bool> getTarget();
	vector<string> getDecision();
	vector<vector<double> > getWeight2D();
	vector<vector<bool> > getDir2D();
	//vector<vector<double> > getThreshold2D();
	vector<double> getWeight();
	vector<bool> getDir();
	vector<bool> getObserve();
	vector<bool> getObserveOld();
	vector<double> getMeasurable();
	vector<double> getMeasurable_old();
	vector<vector<bool> > getMask_amper2D();
	vector<bool> getMask_amper();
	vector<bool> getMask();
	vector<bool> getUp();
	vector<bool> getDown();
	double get_delta_weight_sum(vector<bool> signal);
	/*
	---------------------GET FUNCTION----------------------
	*/

	/*
	---------------------SET FUNCTION----------------------
	*/
	void setTarget(vector<bool> signal);
	void setSignal(vector<bool> observe);
	void setLastTotal(double last_total);
	/*
	---------------------SET FUNCTION----------------------
	*/

	void generate_delayed_weights(vector<vector<double> > &weights, vector<double> &measurable, vector<double> &measurable_old, double last_total, int measurable_id, bool merge, vector<bool> &mask_amper, int LOG_TYPE);
	void amper(vector<vector<bool> > list, int LOG_TYPE);
	void delay(vector<vector<bool> > list);
	void amperand(vector<vector<double> > &weights, vector<double> &measurable, vector<double> &measurable_old, int idx1, int idx2, double total, bool merge, int LOT_TYPE);
	void pruning(vector<bool> signal);
	bool get_delay_amper(int idx, vector<bool> &mask_amper);

	void init_sensors_name(vector<string> &sensors_names,int P_TYPE);
	void init_size(int sensor_size, int P_TYPE);
	void init_direction(int P_TYPE);
	void init_weight(int P_TYPE);
	void init_thresholds(int P_TYPE);
	void init_mask_amper(int P_TYPE);
	void init_other_parameter(int LOG_TYPE);
	void init_name(string name, int P_TYPE);
	void reallocate_memory(int target_sensor_size, int P_TYPE);

	void copy_weight(int P_TYPE, int start_idx, int end_idx, vector<double> &measurable, vector<double> &measurable_old, vector<vector<double> > &weights, vector<vector<bool> > &mask_amper);
	void copy_dir(int P_TYPE, vector<bool> dir);
	void copy_mask(int P_TYPE, vector<bool> mask);
	void copy_mask_amper(int P_TYPE, vector<bool> mask_amper);
	void copy_current(int P_TYPE, vector<bool> current);

	void reset_time(int n, int P_TYPE);
	void reset_base_sensor_size(int base_sensor_size, int P_TYPE);

	void gen_direction(int P_TYPE);
	void gen_weight(int P_TYPE);
	void gen_thresholds(int P_TYPE);
	void gen_mask_amper(int P_TYPE);
	
	void gen_other_parameters(int P_TYPE);
	//void initNewSensor(vector<vector<int> >&list);
	logging get_log();
	void savingData(string filename);
};

class Agent_Stationary: public Agent{
public:
	Agent_Stationary(double q, bool print_info);
	virtual ~Agent_Stationary();
	virtual void update_weights(bool active);
	virtual void orient_all();
	virtual void calculate_total(bool active);
	virtual void calculate_target();

protected:
	//double q;
};


class Agent_Forgetful: public Agent{
public:
	Agent_Forgetful(double q);
	virtual ~Agent_Forgetful();
	virtual void update_weights(bool active);
	virtual void orient_all();
	virtual void calculate_total(bool active);
	virtual void calculate_target();

protected:
	//double q;
};
#endif
