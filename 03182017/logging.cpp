#include "logging.h"

using namespace std;

long long logging::GPU_MEM = 0;
long long logging::CPU_MEM = 0;
int logging::num_sim = 0;

logging::logging(string name){
	this->using_log = using_log;
	this->agent_name = name;
	reset_all();
	init_Type_to_String();
}

logging::~logging(){}

void logging::init_Type_to_String(){
	Type_to_String[logging::PROCESS] = &this->str_process;
	Type_to_String[logging::INIT] = &this->str_init;
	Type_to_String[logging::SIZE] = &this->str_size;

	Type_to_String[logging::MALLOC] = &this->str_malloc;
	Type_to_String[logging::REMALLOC] = &this->str_remalloc;

	Type_to_String[logging::ADD_SENSOR] = &this->str_add_sensor;

	Type_to_String[logging::AMPER] = &this->str_amper;
	Type_to_String[logging::DELAY] = &this->str_delay;
	Type_to_String[logging::AMPERAND] = &this->str_amperand;
	Type_to_String[logging::GENERATE_DELAY] = &this->str_generate_delay;

	//Type_to_String[logging::UP] = &this->str_up_GPU;
	//Type_to_String[logging::SAVING] = &this->str_saving;
	//Type_to_String[logging::LOADING] = &this->str_loading;

	//Type_to_String[logging::COPY] = &this->str_copy;
}

void logging::reset_all(){
	t_halucinate = 0;
	t_orient_all = 0;
	t_propagation = 0;
	t_update_weight = 0;
	n_halucinate = 0;
	n_orient_all = 0;
	n_propagation = 0;
	n_update_weight = 0;
	indent_level = 0;

	str_init = "";
	str_size = "";
	str_add_sensor = "";
	str_up_GPU = "";
	str_saving = "";
	str_loading = "";
	str_num_sim = "";
	str_process = "";
	str_malloc = "";
	str_remalloc = "";
	str_copy = "";
	
	str_amper = "";
	str_delay = "";
	str_amperand = "";
	str_generate_delay = "";

	createEvent();
}

void logging::finalize_stats(perf_stats &stats, int n, double acc_t){
	stats.n = n;
	stats.acc_t = acc_t;
	stats.avg_t = acc_t / n;
}

void logging::finalize_log(){
	finalize_stats(STAT_UPDATE_WEIGHT, n_update_weight, t_update_weight);
	finalize_stats(STAT_ORIENT_ALL, n_orient_all, t_orient_all);
	finalize_stats(STAT_PROPAGATION, n_propagation, t_propagation);
	str_num_sim += "Total Number of Simulation: "+to_string(num_sim)+"\n";
}

void logging::createEvent(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void logging::add_indent(){
	this->indent_level++;
}

void logging::reduce_indent(){
	this->indent_level--;
	if(indent_level < 0) indent_level = 0;
}

void logging::append_log(int LOG_TYPE, string info){
	for(int i = 0; i < indent_level; ++i) info = "    " + info;
	*Type_to_String[LOG_TYPE] += info;
}

void logging::append_process(int LOG_FROM, int LOG_TO){
	*Type_to_String[LOG_TO] += *Type_to_String[LOG_FROM];
	*Type_to_String[LOG_FROM] = "";
}

void logging::add_GPU_MEM(int mem){
	GPU_MEM += mem;
}

void logging::add_CPU_MEM(int mem){
	CPU_MEM += mem;
}

/*
This function start record cuda events, only using_log is true will record event
Input: None
Output: None
*/
void logging::record_start(){
	if(!using_log) return;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
}


/*
This function stop record cuda events, only using_log is true will record event
The time used is calculated and added to the corresponding statistics
Input: None
Output: None
*/
void logging::record_stop(int LOG_TYPE){
	if(!using_log) return;
	float dt = 0;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dt, start, stop);
	switch(LOG_TYPE){
	case logging::UPDATE_WEIGHT:
		n_update_weight++;
		t_update_weight += dt;
		break;
	case logging::ORIENT_ALL:
		n_orient_all++;
		t_orient_all += dt;
		break;
	case logging::PROPAGATION:
		n_propagation++;
		t_propagation += dt;
		break;
	}
}