#include "Agent.h"
#include "logging.h"
#include "AgentDM.h"
/*
----------------Agent Base Class-------------------
*/
extern int ind(int row, int col);
extern int compi(int x);

Agent::Agent(int type, bool using_agent){
	Gdir = NULL;
	Gweights = NULL;
	Gthresholds = NULL;
	GMask_amper = NULL;

	dev_dir = NULL;
	dev_weights = NULL;
	dev_thresholds = NULL;
	dev_mask_amper = NULL;

	this->type = type;
	this->threshold = 1 / 33.0;
	
	this->using_agent = using_agent;
	this->last_total = 0;
	this->total = 0;
	this->memory_expansion = 0.5;
	
	//agentDM=new AgentDM(this);
}

Agent::~Agent(){}

static int randInt(int n){
	return rand()%n;
}

float Agent::decide(vector<bool> signal, double phi, bool active){//the decide function
	this->phi = phi;
	//SIQI:this->q = q;
	setSignal(signal);
	update_state_GPU(active);
	//logging::num_sim++;

	halucinate_GPU();
	return distance_big(dev_load, dev_target);
	//calculate distance

	//TBD
}

vector<string> Agent::getDecision(){
	return decision;
}

void Agent::init_sensors_name(vector<string> &sensors_names, int P_TYPE){
	this->sensors_names = sensors_names;
	//logging_info->append_log(P_TYPE, "Agent Sensors Name From Python, Size: " + to_string(sensors_names.size()) + "\n");
}

void Agent::init_size(int sensor_size, int LOG_TYPE){
	this->sensor_size = sensor_size;
	this->measurable_size = 2 * sensor_size;
	this->array_size = measurable_size * (measurable_size + 1) / 2;
	this->mask_amper_size = sensor_size * (sensor_size + 1);
	
	this->sensor_size_max = (int)(this->sensor_size * (1 + this->memory_expansion));
	this->measurable_size_max = 2 * sensor_size_max;
	this->array_size_max = measurable_size_max * (measurable_size_max + 1) / 2;
	this->mask_amper_size_max = this->sensor_size_max * (this->sensor_size_max + 1);
	
	logging_info->append_log(logging::SIZE, "[SIZE]:\n");
	logging_info->add_indent();

	logging_info->append_log(logging::SIZE, "Sensor Size: " + to_string(this->sensor_size) +
		"(Max Size:" + to_string(this->sensor_size_max) + ")\n");
	logging_info->append_log(logging::SIZE, "Measurable Size: " + to_string(this->measurable_size) +
		"(Max Size:" + to_string(this->measurable_size_max) + ")\n");
	logging_info->append_log(logging::SIZE, "Array Size: " + to_string(this->array_size) +
		"(Max Size:" + to_string(this->array_size_max) + ")\n");
	logging_info->append_log(logging::SIZE, "Mask Size: " + to_string(this->mask_amper_size) +
		"(Max Size:" + to_string(this->mask_amper_size_max) + ")\n");

	logging_info->append_process(logging::SIZE, LOG_TYPE);
	logging_info->reduce_indent();
}

void Agent::init_name(string name, int LOG_TYPE){
	this->name = name;
	logging_info->append_log(LOG_TYPE, "Agent Name From Python: " + name+"\n");
}

void Agent::reset_time(int n, int LOG_TYPE){
	this->t = n;
	logging_info->append_log(LOG_TYPE, "Time Reset to: " + to_string(n) + "\n");
	srand(time(NULL));
	logging_info->append_log(LOG_TYPE, "Time Seed Set \n");
}

void Agent::reset_base_sensor_size(int base_sensor_size, int LOG_TYPE){
	this->base_sensor_size = base_sensor_size;
}

void Agent::init_data(string name, int sensor_size, vector<string> sensors_names, string filename){
	//data init
	this->logging_info = new logging(name);
	//will put this into python as mix object in the end, should be an object across border
	logging_info->append_log(logging::INIT, "[Data Initialization]:\n");
	logging_info->add_indent();
	
	reset_time(0, logging::INIT);
	//cudaCheckErrors("error");
	logging_info->append_log(logging::INIT, "Agent Type : DISCOUNTED\n");
	init_name(name, logging::INIT);
	this->base_sensor_size = sensor_size;
	init_size(sensor_size, logging::INIT);
	//copy_sensors_name(sensors_names, logging::INIT);
	
	//logging_info->append_log(logging::MALLOC,"[Matrix Allocation:]\n");
	//logging_info->add_indent();
	//cudaCheckErrors("error");
	gen_weight(logging::MALLOC);
	//cudaCheckErrors("error");
	gen_direction(logging::MALLOC);
	gen_thresholds(logging::MALLOC);
	gen_mask_amper(logging::MALLOC);
	gen_other_parameters(logging::INIT);
	//logging_info->append_log(logging::COPY, "[Matrix Copy]:\n");
	//logging_info->add_indent();
	init_weight(logging::COPY);
	init_direction(logging::COPY);
	init_thresholds(logging::COPY);
	init_mask_amper(logging::COPY);
	init_other_parameter(logging::MALLOC);
	//logging_info->append_process(logging::COPY, logging::INIT);
	//logging_info->reduce_indent();
	//for now do not save worker infomation
	logging_info->append_process(logging::INIT, logging::PROCESS);
	logging_info->reduce_indent();
}

//those three functions down there are get functions for the variable in C++
vector<bool> Agent::getCurrent(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gcurrent[i]);
	}
	return result;
}

vector<bool> Agent::getPrediction(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gprediction[i]);
	}
	return result;
}

vector<bool> Agent::getSignal(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gsignal[i]);
	}
	return result;
}

vector<bool> Agent::getLoad(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gload[i]);
	}
	return result;
}

logging Agent::get_log(){
	logging_info->finalize_log();
	return *logging_info;
}

void Agent::savingData(string filename){
	agentDM->writeData(filename);
}

void Agent::reallocate_memory(int target_sensor_size, int LOG_TYPE){
	this->logging_info->append_log(logging::REMALLOC, "[REMALLOC:]\n");
	this->logging_info->add_indent();

	free_other_parameters();
	init_size(target_sensor_size, logging::REMALLOC);

	gen_weight(logging::REMALLOC);
	gen_direction(logging::REMALLOC);
	gen_thresholds(logging::REMALLOC);
	gen_mask_amper(logging::REMALLOC);
	gen_other_parameters(logging::REMALLOC);

	init_weight(logging::REMALLOC);
	init_direction(logging::REMALLOC);
	init_thresholds(logging::REMALLOC);
	init_mask_amper(logging::REMALLOC);
	init_other_parameter(logging::REMALLOC);

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::REMALLOC, LOG_TYPE);
}

void Agent::copy_weight(int P_TYPE, int start_idx, int end_idx, vector<double> &measurable, vector<double> &measurable_old, vector<vector<double> > &weights, vector<vector<bool> > &mask_amper){
	int measurable_start = start_idx, measurable_end = end_idx;
	int weight_start = measurable_start * (measurable_start + 1) / 2;
	int weight_end = measurable_end * (measurable_end + 1) / 2;
	int amper_start = (start_idx / 2) * (start_idx / 2 + 1);
	int amper_end = (end_idx / 2) * (end_idx / 2 + 1);
	if(measurable_end > measurable_size_max){
		//record error message in log
		return;
	}
	for(int i = measurable_start; i < measurable_end; ++i){
		GMeasurable[i] = measurable[i];
		GMeasurable_old[i] = measurable_old[i];
	}
	cudaMemcpy(dev_measurable + measurable_start, GMeasurable + measurable_start, (measurable_end - measurable_start) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_measurable_old + measurable_start, GMeasurable_old + measurable_start, (measurable_end - measurable_start) * sizeof(double), cudaMemcpyHostToDevice);

	int x = 0, y = measurable_start;
	for(int i = weight_start; i < weight_end; ++i){
		Gweights[i] = weights[y][x++];
		if(x > y){
			x = 0;
			y++;
		}
	}
	cudaMemcpy(dev_weights + weight_start, Gweights + weight_start, (weight_end - weight_start) * sizeof(double), cudaMemcpyHostToDevice);
	
	x = 0; y = start_idx / 2;
	for(int i = amper_start; i < amper_end; ++i){
		GMask_amper[i] = mask_amper[y][x++];
		if(x > 2 * y + 1){
			x = 0;
			y++;
		}
	}
	cudaMemcpy(dev_mask_amper + amper_start, GMask_amper + amper_start, (amper_end - amper_start) * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::copy_dir(int P_TYPE, vector<bool> dir){
	for(int i = 0; i < this->array_size; ++i) Gdir[i] = dir[i];
	cudaMemcpy(dev_dir, Gdir, this->array_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::copy_mask(int P_TYPE, vector<bool> mask){
	for(int i = 0; i < mask.size(); ++i) Gmask[i] = mask[i];
	cudaMemcpy(dev_mask, Gmask, mask.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::copy_mask_amper(int P_TYPE, vector<bool> mask_amper){
	for(int i = 0; i < mask_amper.size(); ++i) GMask_amper[i] = mask_amper[i];
	cudaMemcpy(dev_mask_amper, GMask_amper, this->mask_amper_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::copy_current(int P_TYPE, vector<bool> current){
	for(int i = 0; i < current.size(); ++i) Gcurrent[i] = current[i];
	cudaMemcpy(dev_current, Gcurrent, current.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::pruning(vector<bool> signal){
	vector<vector<double> > weights = getWeight();
	vector<double> measurable = getMeasurable();
	vector<double> measurable_old = getMeasurable_old();
	vector<vector<bool> > mask_amper = getMask_amper();

	//logging
	//pruning for meaurable, measurable_old and weights
	int col_escape = 0;
	for(int i = 0; i < weights.size(); ++i){
		if(signal[i]) col_escape++;//need to escape row
		else{
			int row_escape = 0;
			for(int j = 0; j < weights[i].size(); ++j){
				if(signal[j]) row_escape++;//itself need to be removed
				else weights[i - col_escape][j - row_escape] = weights[i][j];
			}
			measurable[i - col_escape] = measurable[i];
			measurable_old[i - col_escape] = measurable_old[i];

			for(int j = 0; j < row_escape; ++j) weights[i].pop_back();
		}
	}
	for(int i = 0; i < col_escape; ++i){
		weights.pop_back();
		measurable.pop_back();
		measurable_old.pop_back();
	}
	//pruning for mask_amper
	col_escape = 0;
	for(int i = 0; i < mask_amper.size(); ++i){
		if(signal[2 * i]) col_escape++;
		else{
			int row_escape = 0;
			for(int j = 0; j < mask_amper[i].size(); ++j){
				if(signal[j]) row_escape++;//itself need to be removed
				else weights[i - col_escape][j - row_escape] = weights[i][j];
			}

			for(int j = 0; j < row_escape; ++j) weights[i].pop_back();
		}
	}
	for(int i = 0; i < col_escape; ++i){
		mask_amper.pop_back();
	}
	//write back to GPU and change matrix size
	copy_weight(logging::INIT, measurable_size, measurable.size(), measurable, measurable_old, weights, mask_amper);
	this->sensor_size = measurable.size() / 2;
	this->measurable_size = measurable.size();
	this->array_size = this->measurable_size * (this->measurable_size + 1) / 2;
	this->mask_amper_size = this->sensor_size * (this->sensor_size + 1);
}

void Agent::amper(vector<vector<bool> > lists, int LOG_TYPE){
	vector<vector<double> > weights = getWeight();
	vector<double> measurable = getMeasurable();
	vector<double> measurable_old = getMeasurable_old();
	vector<vector<bool> > mask_amper = getMask_amper();

	this->logging_info->append_log(logging::AMPER, "[AMPER]:\n");

	for(int i = 0; i < lists.size(); ++i){
		vector<int> list;
		for(int j = 0; j < lists[i].size(); ++j){
			if(lists[i][j]) list.push_back(j);
		}
		if(list.size() < 2) continue;//probably need sth to append in error.log
		amperand(weights, measurable, measurable_old, list[0], list[1], total, true, LOG_TYPE);
		for(int j = 2; j < list.size(); ++j){
			amperand(weights, measurable, measurable_old, list[j], measurable.size() - 2, total, false, LOG_TYPE);
		}

		int s = mask_amper.back().size() + 2;
		vector<bool> tmp;
		for(int j = 0; j < s; ++j){
			tmp.push_back(lists[i][j]);
		}
		mask_amper.push_back(tmp);
	}
	
	if(measurable.size() > measurable_size_max){//need to reallocate
		reallocate_memory(measurable.size() / 2, logging::AMPER);
		copy_weight(logging::INIT, 0, measurable.size(), measurable, measurable_old, weights, mask_amper);
	}
	else{//just copy to the back
		copy_weight(logging::INIT, measurable_size, measurable.size(), measurable, measurable_old, weights, mask_amper);
		this->sensor_size = measurable.size() / 2;
		this->measurable_size = measurable.size();
		this->array_size = this->measurable_size * (this->measurable_size + 1) / 2;
		this->mask_amper_size = this->sensor_size * (this->sensor_size + 1);
	}
}

void Agent::delay(vector<vector<bool> > lists){
	vector<vector<double> > weights = getWeight();
	vector<double> measurable = getMeasurable();
	vector<double> measurable_old = getMeasurable_old();
	vector<vector<bool> > mask_amper = getMask_amper();
	this->logging_info->append_log(logging::DELAY, "[DELAY]:(t=" + to_string(this->t) + ")\n");
	this->logging_info->add_indent();
	int success_amper = 0;

	this->logging_info->append_log(logging::DELAY, "Init Measurable Size: " + to_string(measurable.size()) + "\n");
	
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list;
		for(int j = 0; j < lists[i].size(); ++j){
			if(lists[i][j]) list.push_back(j);
		}
		if(list.size() == 0) continue;//probably need sth to append in error.log
		success_amper++;
		
		if(list.size() != 1){
			amperand(weights, measurable, measurable_old, list[0], list[1], total, true, logging::DELAY);
			for(int j = 2; j < list.size(); ++j){
				amperand(weights, measurable, measurable_old, list[j], measurable.size() - 2, total, false, logging::DELAY);
			}
			generate_delayed_weights(weights, measurable, measurable_old, last_total, weights.size() - 2, false, logging::DELAY);
		}
		else{
			generate_delayed_weights(weights, measurable, measurable_old, last_total, list[0], true, logging::DELAY);
		}
		
		int s = mask_amper.back().size() + 2;
		vector<bool> tmp;
		for(int j = 0; j < s; ++j){
			tmp.push_back(lists[i][j]);
		}
		mask_amper.push_back(tmp);
	}

	this->logging_info->append_log(logging::DELAY, "Amper Done: " + to_string(success_amper) + "("+ to_string(lists.size()) + ")\n");

	if(measurable.size() > measurable_size_max){//need to reallocate
		reallocate_memory(measurable.size() / 2, logging::DELAY);
		copy_weight(logging::INIT, 0, measurable.size(), measurable, measurable_old, weights, mask_amper);
	}
	else{//just copy to the back
		copy_weight(logging::INIT, measurable_size, measurable.size(), measurable, measurable_old, weights, mask_amper);
		this->sensor_size = measurable.size() / 2;
		this->measurable_size = measurable.size();
		this->array_size = this->measurable_size * (this->measurable_size + 1) / 2;
		this->mask_amper_size = this->sensor_size * (this->sensor_size + 1);
	}

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::DELAY, logging::PROCESS);
}

/*
This function generate direction matrix
It will first delete the old Matrix
And then generate new one
Log info is saved
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_direction(int P_TYPE){
	delete[] Gdir;

	Gdir = new bool[array_size_max];
	//logging_info->append_log(P_TYPE, "Direction Matrix Generated, Size of Matrix: "
		//+ to_string(measurable_size) + "*"+to_string(measurable_size) + "\n");
	//logging::add_CPU_MEM(measurable_size * measurable_size * sizeof(bool));

	cudaFree(dev_dir);

	cudaMalloc(&dev_dir, array_size_max * sizeof(bool));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Direction Matrix: "
		//+ to_string(measurable_size * measurable_size * sizeof(bool)) + " Bytes(" + to_string
		//(measurable_size * measurable_size * sizeof(bool) / (1024.0 * 1024)) + "M)\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(bool));
}

/*
This function generate weight matrix
It will first delete the old Matrix
And then generate new one
Log info is saved
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_weight(int P_TYPE){
	delete[] Gweights;

	Gweights = new double[array_size_max];
	//logging_info->append_log(P_TYPE, "Weight Matrix Generated, Size of Matrix: "
		//+ to_string(measurable_size) + "*" + to_string(measurable_size) + "\n");
	//logging::add_CPU_MEM(measurable_size * measurable_size * sizeof(double));
	
	cudaFree(dev_weights);

	cudaMalloc(&dev_weights, array_size_max * sizeof(double));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Weight Matrix: "
		//+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		//(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(double));
}

/*
This function generate thresholds matrix
It will first delete the old Matrix
And then generate new one
Log info is saved
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_thresholds(int P_TYPE){
	delete[] Gthresholds;

	Gthresholds = new double[array_size_max];
	//logging_info->append_log(P_TYPE, "Thresholds Matrix Generated, Size of Matrix: "
		//+ to_string(measurable_size) + "*" + to_string(measurable_size) + "\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(double));
	
	cudaFree(dev_thresholds);

	cudaMalloc(&dev_thresholds, array_size_max * sizeof(double));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Thresholds Matrix: "
		//+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		//(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(double));
}

void Agent::gen_mask_amper(int P_TYPE){
	delete[] GMask_amper;

	GMask_amper = new bool[mask_amper_size_max];

	cudaFree(dev_mask_amper);

	cudaMalloc(&dev_mask_amper, mask_amper_size_max * sizeof(bool));
}

/*
This function copy direction matrix from CPU to GPU.
If no data is provided, use default method, otherwise use the data value
Input: data to copy to GPU, log type, indicating where this function is invoked
Output: None
*/
void Agent::init_direction(int P_TYPE){
	int x = 0, y = 0;
	for(int i = 0; i < array_size; ++i){
		Gdir[i] = (x == y);
		x++;
		if(x > y){
			y++;
			x = 0;
		}
	}
	cudaMemcpy(dev_dir, Gdir, array_size * sizeof(bool), cudaMemcpyHostToDevice);
	//logging_info->append_log(P_TYPE, "Direction Matrix Copied to GPU\n");
}

/*
This function copy weight matrix from CPU to GPU.
If no data is provided, use default method(=0.0), otherwise use the data value
Input: data to copy to GPU, log type, indicating where this function is invoked
Output: None
*/
void Agent::init_weight(int P_TYPE){
	for(int i = 0; i < array_size; ++i){
		Gweights[i] = 0.0;
	}
	cudaMemcpy(dev_weights, Gweights, array_size * sizeof(double), cudaMemcpyHostToDevice);
	//logging_info->append_log(P_TYPE, "Weight Matrix Copied to GPU\n");
}

/*
This function copy thresholds matrix from CPU to GPU.
If no data is provided, use default method(=threshold), otherwise use the data value
Input: data to copy to GPU, log type, indicating where this function is invoked
Output: None
*/
void Agent::init_thresholds(int P_TYPE){
	for(int i = 0; i < array_size; ++i){
		Gdir[i] = threshold;
	}
	cudaMemcpy(dev_thresholds, Gthresholds, array_size * sizeof(double), cudaMemcpyHostToDevice);
	//logging_info->append_log(P_TYPE, "Thresholds Matrix Copied to GPU\n");
}

void Agent::init_mask_amper(int P_TYPE){
	cudaMemset(dev_mask_amper, false, mask_amper_size_max * sizeof(bool));
}

void Agent::init_other_parameter(int LOG_TYPE){
	cudaMemset(dev_observe, 0, measurable_size_max * sizeof(bool));
	cudaMemset(dev_signal, false, measurable_size_max * sizeof(bool));
	cudaMemset(dev_load, false, measurable_size_max * sizeof(bool));
	cudaMemset(dev_mask, false, measurable_size_max * sizeof(bool));
	cudaMemset(dev_current, false, measurable_size_max * sizeof(bool));
	cudaMemset(dev_target, false, measurable_size_max * sizeof(bool));
	
	cudaMemset(dev_signal1, false, measurable_size_max * sizeof(bool));
	cudaMemset(dev_signal2, false, measurable_size_max * sizeof(bool));
	cudaMemset(dev_measurable, 0, measurable_size_max * sizeof(double));
	cudaMemset(dev_measurable_old, 0, measurable_size_max * sizeof(double));
}

/*
This function generate other parameter
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_other_parameters(int P_TYPE){
	Gobserve = new bool[measurable_size_max];
	Gsignal = new bool[measurable_size_max];
	Gload = new bool[measurable_size_max];
	Gmask = new bool[measurable_size_max];
	Gcurrent = new bool[measurable_size_max];
	Gtarget = new bool[measurable_size_max];
	GMeasurable = new double[measurable_size_max];
	GMeasurable_old = new double[measurable_size_max];
	Gprediction = new bool[measurable_size_max];
	Gup = new bool[measurable_size_max];
	Gdown = new bool[measurable_size_max];
	//logging_info->append_log(P_TYPE, "Other Parameter Generated\n");
	//logging::add_CPU_MEM(5 * whole_size * sizeof(bool) + workerSize * sizeof(bool));
	
	cudaMalloc(&dev_observe, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_signal, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_load, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_mask, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_current, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_target, measurable_size_max * sizeof(bool));
	
	cudaMalloc(&dev_signal1, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_signal2, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_measurable, measurable_size_max * sizeof(double));
	cudaMalloc(&dev_measurable_old, measurable_size_max * sizeof(double));
	//cudaMalloc(&dev_sensor_value, whole_size * sizeof(float));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Other Parameter: " +
		//to_string(5 * whole_size * sizeof(bool) + workerSize * sizeof(bool) + whole_size * sizeof(float)) + "\n");
	//logging::add_GPU_MEM(5 * whole_size * sizeof(bool) + workerSize * sizeof(bool));
}

/*
------------------------------------SET FUNCTION------------------------------------
*/
void Agent::setTarget(vector<bool> signal){
	for(int i = 0; i < measurable_size; ++i){
		Gtarget[i] = signal[i];
	}
	cudaMemcpy(dev_target, Gtarget, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}
/*
------------------------------------SET FUNCTION------------------------------------
*/

/*
------------------------------------GET FUNCTION------------------------------------
*/
vector<bool> Agent::getTarget(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gtarget[i]);
	}
	return result;
}

vector<vector<double> > Agent::getWeight(){
	cudaMemcpy(Gweights, dev_weights, array_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<vector<double> > result;
	int n = 0;
	for(int i = 0; i < measurable_size; ++i){
		vector<double> tmp;
		for(int j = 0; j <= i; ++j)
			tmp.push_back(Gweights[n++]);
		result.push_back(tmp);
	}
	return result;
}

vector<vector<bool> > Agent::getDir(){
	cudaMemcpy(Gdir, dev_dir, array_size * sizeof(bool), cudaMemcpyDeviceToHost);
	vector<vector<bool> > result;
	int n = 0;
	for(int i = 0; i < measurable_size; ++i){
		vector<bool> tmp;
		for(int j = 0; j <= i; ++j)
			tmp.push_back(Gdir[n++]);
		result.push_back(tmp);
	}
	return result;
}

vector<double> Agent::getMeasurable(){
	cudaMemcpy(GMeasurable, dev_measurable, measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(GMeasurable[i]);
	}
	return result;
}

vector<double> Agent::getMeasurable_old(){
	cudaMemcpy(GMeasurable_old, dev_measurable_old, measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(GMeasurable_old[i]);
	}
	return result;
}

vector<vector<bool> > Agent::getMask_amper(){
	vector<vector<bool> > result;
	cudaMemcpy(GMask_amper, dev_mask_amper, mask_amper_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int n = 0;
	for(int i = 0; i < this->sensor_size; ++i){
		vector<bool> tmp;
		for(int j = 0; j <= 2 * i + 1; ++j)
			tmp.push_back(GMask_amper[n++]);
		result.push_back(tmp);
	}
	return result;
}

vector<bool> Agent::getMask(){
	vector<bool> result;
	cudaMemcpy(Gmask, dev_mask, this->measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < this->measurable_size; ++i) result.push_back(Gmask[i]);
	return result;
}

vector<bool> Agent::getObserve(){
	vector<bool> result;
	cudaMemcpy(Gobserve, dev_observe, this->measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < this->measurable_size; ++i) result.push_back(Gobserve[i]);
	return result;
}

vector<bool> Agent::getUp(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gup[i]);
	}
	return result;
}

vector<bool> Agent::getDown(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gdown[i]);
	}
	return result;
}
/*
------------------------------------GET FUNCTION------------------------------------
*/

void Agent::amperand(vector<vector<double> > &weights, vector<double> &measurable, vector<double> &measurable_old, int idx1, int idx2, double total, bool merge, int LOG_TYPE){
	vector<double> result1, result2;

	this->logging_info->append_log(logging::AMPERAND, "[AMPERAND]:\n");
	this->logging_info->add_indent();
	
	this->logging_info->append_log(logging::AMPERAND, "Amper Index: " + to_string(idx1) + ", " + to_string(idx2) + "\n");

	for(int i = 0; i < measurable.size(); ++i){
		if(i == idx1 || i == idx2){
			result1.push_back(weights[idx2][idx1]);
			result1.push_back(0);
		}
		else{
			result1.push_back(weights[idx2][idx1] * measurable[i] / total);
			result1.push_back(weights[idx2][idx1] * (1 - measurable[i] / total));
		}
	}
	result1.push_back(result1[0] + result1[1]);
	
	//first row
	for(int i = 0; i < measurable.size(); ++i){
		if(i == idx1){
			result2.push_back(weights[compi(idx2)][idx1]);
			result2.push_back(weights[idx2][compi(idx1)] + weights[compi(idx2)][compi(idx1)]);
		}
		else if(i == idx2){
			result2.push_back(weights[idx2][compi(idx1)]);
			result2.push_back(weights[compi(idx2)][idx1] + weights[compi(idx2)][compi(idx1)]);
		}
		else{
			result2.push_back((1 - weights[idx2][idx1]) * measurable[i] / total);
			result2.push_back((1 - weights[idx2][idx1]) * (1 - measurable[i] / total));
		}
	}
	result2.push_back(0);
	result2.push_back(result2[0] + result2[1]);

	if(merge){//if it is the new row that need to append without pop
		weights.push_back(result1);
		weights.push_back(result2);
		measurable.push_back(result1.back());
		measurable.push_back(result2.back());
		measurable_old.push_back(result1.back());
		measurable_old.push_back(result2.back());

		this->logging_info->append_log(logging::AMPERAND, "New Sensor Merged in Temp Variable, Measurable Size: " + to_string(measurable.size()) + ")\n");
		//just put in two new vector and two cc value
	}
	else{//else need to pop the old value and push in the new one, need t get rid of corresponding value
		int old_idx1 = measurable.size() - 2, old_idx2 = measurable.size() - 1;
		result1.erase(result1.begin() + old_idx1);
		result1.erase(result1.begin() + old_idx2);
		result2.erase(result2.begin() + old_idx1);
		result2.erase(result2.begin() + old_idx2);
		weights.pop_back();weights.pop_back();
		weights.push_back(result1);
		weights.push_back(result2);
		measurable.pop_back();measurable.pop_back();
		measurable_old.pop_back();measurable_old.pop_back();
		measurable.push_back(result1.back());
		measurable.push_back(result2.back());
		measurable_old.push_back(result1.back());
		measurable_old.push_back(result2.back());

		this->logging_info->append_log(logging::AMPERAND, "New Sensor Replace in Temp Variable, Measurable Size: " + to_string(measurable.size()) + ")\n");
	}
	//second row

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::AMPERAND, LOG_TYPE);
}

void Agent::generate_delayed_weights(vector<vector<double> > &weights, vector<double> &measurable, vector<double> &measurable_old, double last_total, int measurable_id, bool merge, int LOG_TYPE){
	vector<double> result1, result2;

	this->logging_info->append_log(logging::GENERATE_DELAY, "[GENERATE DELAY SENSOR]:\n");
	this->logging_info->add_indent();

	int delay_sensor_id1 = measurable_id, delay_sensor_id2 = compi(measurable_id);

	this->logging_info->append_log(logging::GENERATE_DELAY, "Delay Index: " + to_string(delay_sensor_id1) + ", " + to_string(delay_sensor_id2) + "\n");

	for(int i = 0; i < measurable.size() - 2 + 2 * merge; ++i){
		result1.push_back(measurable_old[delay_sensor_id1] * measurable[i] / last_total);
	}
	result1.push_back(result1[0] + result1[1]);
	//row 1
	for(int i = 0; i < measurable.size() - 2 + 2 * merge; ++i){
		result2.push_back(measurable_old[delay_sensor_id2] * measurable[i] / last_total);
	}
	result2.push_back(0);
	result2.push_back(result2[0] + result2[1]);
	//row 2

	if(!merge){
		weights.pop_back();weights.pop_back();
	}
	weights.push_back(result1);
	weights.push_back(result2);

	if(!merge){
		measurable.pop_back();measurable.pop_back();
		measurable_old.pop_back();measurable_old.pop_back();
	}
	measurable.push_back(result1.back());
	measurable.push_back(result2.back());
	measurable_old.push_back(result1.back());
	measurable_old.push_back(result2.back());

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::GENERATE_DELAY, LOG_TYPE);
}

/*
This function update state on GPU, it is the main function for simulation on C++
It contains three main parts: update weight, orient all, propagation, result will be stored in Gload(propagate_GPU)
Input: mode to use
Output: None
*/
void Agent::update_state_GPU(bool activity){//true for decide	
    // udpate the snapshot weights and total count:
    update_weights(activity);
	calculate_total(activity);

	// compute the derived orientation matrix and update the thresholds:
	orient_all();

	//SIQI:here I have intervened to disconnect the automatic computation of a target. Instead, I will be setting the target externally (from the Python side) at the beginning of each state-update cycle.
	// compute the target state:
	// calculate_target();

	cudaMemcpy(dev_signal, dev_observe, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load, false);

	cudaMemcpy(Gcurrent, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(Gdir, dev_dir, whole_size * whole_size * sizeof(bool), cudaMemcpyDeviceToHost);

	t++;
}

/*
This function is halucinate on GPU, it use several propagate_GPU
It first get the mask to use and then use propagate
The output will be stored in Gload(propagate_GPU)
Input: action list to be halucinated
Output: None
*/
void Agent::halucinate_GPU(){
	gen_mask();

	cudaMemcpy(dev_signal, dev_mask, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(dev_load, dev_current, whole_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load, false);
	cudaMemcpy(Gload, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(Gprediction, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function set signal for simulation in each frame
Input: observe value from python side
Output: None
*/
void Agent::setSignal(vector<bool> observe){//this is where data comes in in every frame
	for(int i = 0; i < observe.size(); ++i){
		Gobserve[i] = observe[i];
	}
	cudaMemcpy(dev_observe, Gobserve, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
This function free data together
Deprecated in new version
Input: None
Output: None
*/
void Agent::free_other_parameters(){//free data in case of memory leak
	delete[] Gobserve;
	delete[] Gsignal;
	delete[] Gload;
	delete[] Gmask;
	delete[] Gcurrent;
	delete[] Gtarget;
	delete[] GMeasurable;
	delete[] GMeasurable_old;
	delete[] Gprediction;
	delete[] Gup;
	delete[] Gdown;

	cudaFree(dev_mask);
	cudaFree(dev_current);
	cudaFree(dev_target);

	cudaFree(dev_observe);
	cudaFree(dev_signal);
	cudaFree(dev_load);
	
	cudaFree(dev_signal1);
	cudaFree(dev_signal2);
	cudaFree(dev_measurable);
	cudaFree(dev_measurable_old);
}

void Agent::calculate_total(bool active){}

void Agent::calculate_target(){}

/*
This function is update_weights based function for all types of agents
Input: None
Output: None
*/
void Agent::update_weights(bool active){}

void Agent::orient_all(){}

void Agent::update_thresholds(){}
/*
----------------Agent Base Class-------------------
*/

/*
----------------Agent_Stationary Class-------------------
*/

Agent_Stationary::Agent_Stationary(double q,  bool using_agent)
	:Agent(STATIONARY, using_agent){
	this->q = q;
}

Agent_Stationary::~Agent_Stationary(){}

/*
----------------Agent_Stationary Class-------------------
*/


/*
----------------Agent_Forgetful Class-------------------
*/

Agent_Forgetful::Agent_Forgetful(double q,  bool using_agent)
	:Agent(FORGETFUL, using_agent){
	this->q = q;
}

Agent_Forgetful::~Agent_Forgetful(){}

/*
----------------Agent_Forgetful Class-------------------
*/
