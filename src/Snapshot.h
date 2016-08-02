#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include <vector>
#include <iostream>
#include <map>
#include <time.h>
using namespace std;

class Snapshot{
public:
	Snapshot();
	Snapshot(double threshold);
	virtual ~Snapshot();

	void initData(string name,int size,double threshold,vector<vector<int> > context_key,vector<int> context_value,vector<string> sensors_names,vector<string> evals_names,vector<vector<int> > generalized_actions);
	void freeData();
	void update_state_GPU(bool mode);
	void propagate_GPU();
	void halucinate_GPU(vector<int> actions_list);
	vector<bool> initMask(vector<int> actions_list);
	void setSignal(vector<bool> observe);
	vector<bool> getCurrent();
	vector<bool> getLoad();
	vector<vector<bool> > getDir();
	vector<bool> halucinate(vector<int> action_list);

protected:
	int size;
	double threshold;
	std::map<pair<int,int>,int> context;
	vector<string> sensors_names,evals_names;
	vector<vector<int> > generalized_actions;
	string name;
	std::map<string,int> name_to_num;

private:
	//those values are CPU and GPU counterpart variables. usually in GPU variable start with dev_(device)
	bool *Gdir,*dev_dir;//dir is DIR in python
	double *Gweights,*dev_weights,*Gthresholds,*dev_thresholds;//weight and threshold in python
	bool *Gobserve,*dev_observe;//observe in python
	bool *Gdfs,*dev_dfs;//this variable is a bool value used in dfs function
	bool *Gsignal,*dev_signal,*Gload,*dev_load;//signal and load variable in propagate
	bool *Gcurrent,*dev_current;//current in python
	bool *Gmask,*dev_mask;//bool value for mask signal in halucinate
	int *tmp_signal,*tmp_load;//tmp variable for dfs on GPU, those two variable are mainly used in bool2int and int2bool, which are tricky ways to mark 'visited' in dfs
	int *dev_scan;
	int *tmp_weight;
	// need to add temp_dir for matrix multiplication
	int *tmp_dir;
	int *out_signal, *out_load;
};

#endif