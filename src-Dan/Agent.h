#ifndef _AGENT_
#define _AGENT_

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <time.h>
#include "worker.h"
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
	bool *Gdfs,*dev_dfs;//this variable is a bool value used in dfs function
	bool *Gsignal,*dev_signal,*Gload,*dev_load;//signal and load variable in propagate
	bool *Gcurrent,*dev_current;//current in python
	bool *Gmask,*dev_mask;//bool value for mask signal in halucinate
	int *dev_scan;
	// need to add temp_dir for matrix multiplication
	bool *out_signal, *out_load;
	bool *dev_affected_worker;
	worker *Gworker,*dev_worker;
	float *dev_sensor_value;
	//variables used in kernel.cu

protected:
	//general variables for agent
	int sensorSize,measurableSize,workerSize;
	double threshold;
	string name,message;
	int type;
	std::map<pair<int,int>,int> context;
	vector<string> sensors_names,evals_names,decision;
	vector<vector<int> > generalized_actions;
	std::map<string,int> name_to_num;
	//general variables for agent

public:
	enum agent_type{EMPIRICAL,DISTRIBUTED,DISCOUNTED};

public:
	Agent(int type,double threshold);
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
	void update_state_GPU(bool mode);
	void propagate_GPU();
	void halucinate_GPU(vector<int> actions_list);
	vector<bool> initMask(vector<int> actions_list);
	void setSignal(vector<bool> observe);
	vector<bool> getCurrent();
	vector<bool> getLoad();
	vector<vector<bool> > getDir();
	vector<bool> halucinate(vector<int> action_list);
	virtual void initWorkerMemory(double *weights,bool *dir);
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
	Agent_Empirical(double threshold);
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
	Agent_Distributed(double threshold);
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
	Agent_Discounted(double threshold,double q);
	virtual ~Agent_Discounted();
	virtual void update_weights();
	virtual void initWorkerMemory(double *weights,bool *dir);
};

/*
----------------Agent_Discounted Class-------------------
*/

#endif