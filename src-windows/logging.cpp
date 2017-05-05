#include "logging.h"
#include <fstream>
#include <sstream>

using namespace std;

long long logging::GPU_MEM = 0;
long long logging::CPU_MEM = 0;
int logging::num_sim = 0;

logging::logging(string name, string filter){
	this->agent_name = name;
	init_log_name_to_type();
	init_Type_to_String();
	init_log_section_name();
	read_log_filter(filter);
	reset_all();
}

logging::~logging(){}

void logging::read_log_filter(string file){
	for(int i = PROCESS; i <= GENERATE_DELAY; ++i){
		log_status[i] = 1;
	}
	if(file=="") return;

	std::ifstream infile(file);
	string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		string log;
		int b;
		iss >> log >> b;
		if(log_name_to_type.find(log) !=log_name_to_type.end() && log_status.find(log_name_to_type[log]) != log_status.end()){
			log_status[log_name_to_type[log]] = b;
		}
	}
}

void logging::init_log_name_to_type(){
	log_name_to_type["PROCESS"] = logging::PROCESS;
	log_name_to_type["INIT"] = logging::INIT;
	log_name_to_type["SIZE"] = logging::SIZE;
	log_name_to_type["MALLOC"] = logging::MALLOC;
	log_name_to_type["REMALLOC"] = logging::REMALLOC;
	log_name_to_type["ADD_SENSOR"] = logging::ADD_SENSOR;
	log_name_to_type["AMPER"] = logging::AMPER;
	log_name_to_type["DELAY"] = logging::DELAY;
	log_name_to_type["AMPERAND"] = logging::AMPERAND;
	log_name_to_type["GENERATE_DELAY"] = logging::GENERATE_DELAY;
	log_name_to_type["UP"] = logging::UP;
	log_name_to_type["SAVING"] = logging::SAVING;
	log_name_to_type["LOADING"] = logging::LOADING;
	log_name_to_type["INIT_DATA"] = logging::INIT_DATA;
	log_name_to_type["COPY_DATA"] = logging::COPY_DATA;
}

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

	Type_to_String[logging::UP] = &this->str_up_GPU;
	Type_to_String[logging::SAVING] = &this->str_saving;
	Type_to_String[logging::LOADING] = &this->str_loading;

	Type_to_String[logging::INIT_DATA] = &this->str_init_data;
	Type_to_String[logging::COPY_DATA] = &this->str_copy_data;
}

void logging::init_log_section_name(){
	log_section_name[logging::INIT] = "[Initialization]:\n";
	log_section_name[logging::SIZE] = "[SIZE]:\n";
	log_section_name[logging::ADD_SENSOR] = "\n";
	log_section_name[logging::UP] = "[UP]:\n";
	log_section_name[logging::SAVING] = "[SAVING]:\n";
	log_section_name[logging::LOADING] = "[LOADING]:\n";

	log_section_name[logging::PROCESS] = "";
	log_section_name[logging::MALLOC] = "[MALLOC]:\n";
	log_section_name[logging::REMALLOC] = "[REMALLOC]:\n";
	log_section_name[logging::COPY_DATA] = "[COPY DATA]:\n";
	log_section_name[logging::INIT_DATA] = "[INIT DATA]:\n";
	
	log_section_name[logging::AMPER] = "[AMPER]:\n";
	log_section_name[logging::DELAY] = "[DELAY]:\n";
	log_section_name[logging::AMPERAND] = "[AMPERAND]:\n";
	log_section_name[logging::GENERATE_DELAY] = "[GENERATE DELAY]:\n";
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

	createEvent();

	for(int i = PROCESS; i <= GENERATE_DELAY; ++i){
		*Type_to_String[i] = log_section_name[i];
	}
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
	string log_from = *Type_to_String[LOG_FROM];
	for(int i = 0; i < indent_level - 1; ++i) log_from = "    " + log_from;

	if(log_status[LOG_FROM] && log_status[LOG_TO])
		*Type_to_String[LOG_TO] += log_from;
	*Type_to_String[LOG_FROM] = log_section_name[LOG_FROM];
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

void logging::set_result(string result){
	this->result = result;
}

string logging::get_result(){
	return this->result;
}