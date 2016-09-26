#ifndef _AGENT_
#define _AGENT_

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <time.h>
#include "worker.h"
#include "logging.h"
using namespace std;

/*
----------------Agent Base Class-------------------
*/

class Agent{
protected:
	//variables used in kernel.cu
	bool *Gdir,*dev_dir;//dir is DIR in python
	double *Gweights,*dev_weights,*Gthresholds,*dev_thresholds;//weight and threshold in python
	bool *Gobserve,*dev_observe;//observe in python
	bool *Gsignal,*dev_signal,*Gload,*dev_load;//signal and load variable in propagate
	bool *Gcurrent,*dev_current;//current in python
	bool *Gmask,*dev_mask;//bool value for mask signal in halucinate
	bool *Gaffected_worker,*dev_affected_worker;
	worker *Gworker,*dev_worker;
	float *dev_sensor_value;
	//variables used in kernel.cu

protected:
	//general variables for agent
	int sensorSize,measurableSize,workerSize,t;
	double threshold;
	string name,message;
	int type;
	std::map<pair<int,int>,int> context;
	vector<string> sensors_names,evals_names,decision;
	vector<vector<int> > generalized_actions;
	std::map<string,int> name_to_num;
	bool is_log_on,is_worker_solution;
	vector<vector<bool> > touched_workers,projected_signal;
	logging *logging_info;
	//general variables for agent

public:
	int test;
	enum agent_type{EMPIRICAL,DISTRIBUTED,DISCOUNTED};
	vector<bool> selected_touched_workers,selected_projected_signal;

public:
	Agent(int type,double threshold,bool using_worker,bool using_log);
	virtual ~Agent();
	void decide(string mode,vector<int> param1,string param2);
	vector<string> translate(vector<int> index_list);
	bool checkParam(vector<int> param);
	bool checkParam(string param);

	vector<string> getDecision();
	string getMessage();

	void initData(string name,int sensorSize,vector<vector<int> > context_key,vector<int> context_value,vector<string> sensors_names,vector<string> evals_names,vector<vector<int> > generalized_actions);
	void freeData();
	virtual void update_weights();
	void orient_all();
	void update_state_GPU(bool mode);
	void propagate_GPU();
	void up_GPU(vector<bool> signal);
	void halucinate_GPU(vector<int> &actions_list);
	vector<bool> initMask(vector<int> &actions_list);
	void setSignal(vector<bool> observe);
	vector<bool> getCurrent();
	vector<bool> getSignal();
	vector<bool> getLoad();
	vector<bool> getAffectedWorkers();
	vector<vector<bool> > getDir();
	vector<bool> halucinate(vector<int> &action_list);
	virtual void initWorkerMemory(double *weights,bool *dir);
	void appendSensor(int id1, int id2, vector<vector<double> > &data, bool merge);
	void addSensor(vector<int> &list,vector<vector<double> > &data);
	vector<vector<double> > addSensors(vector<vector<int> > &list);
	vector<vector<double> > getVectorWeight();
	void copy_sensors_name(vector<string> &sensors_names);
	void copy_context(vector<vector<int> > &context_key,vector<int> &context_value);
	void copy_evals_name(vector<string> &evals_names);
	void copy_generalized_actions(vector<vector<int> > &generalized_actions);
	void copy_size(int measurableSize);
	void copy_direction(vector<vector<bool> >&data);
	void copy_weight(vector<vector<double> >&data);
	void copy_thresholds(vector<vector<double> >&data);
	void copy_name(string name);
	void reset_time(int n);
	void gen_worker();
	void gen_other_parameters();
	void initNewSensor(vector<vector<int> >&list);
	logging get_log();
};

/*
----------------Agent Base Class-------------------
*/

/*
----------------Agent_Empirical Class-------------------
*/

class Agent_Empirical:public Agent{
private:
public:
	Agent_Empirical(double threshold,bool using_worker,bool using_log);
	virtual ~Agent_Empirical();
	virtual void update_weights();
	virtual void initWorkerMemory(double *weights,bool *dir);
};

/*
----------------Agent_Empirical Class-------------------
*/

/*
----------------Agent_Distributed Class-------------------
*/

class Agent_Distributed:public Agent{
private:
public:
	Agent_Distributed(double threshold,bool using_worker,bool using_log);
	virtual ~Agent_Distributed();
	virtual void update_weights();
	virtual void initWorkerMemory(double *weights,bool *dir);
};

/*
----------------Agent_Distributed Class-------------------
*/

/*
----------------Agent_Discounted Class-------------------
*/

class Agent_Discounted:public Agent{
private:
	double q;
public:
	Agent_Discounted(double threshold,double q,bool using_worker,bool using_log);
	virtual ~Agent_Discounted();
	virtual void update_weights();
	virtual void initWorkerMemory(double *weights,bool *dir);
};

/*
----------------Agent_Discounted Class-------------------
*/

#endif