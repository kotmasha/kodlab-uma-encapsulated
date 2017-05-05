#ifndef _LOGGING_
#define _LOGGING_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Global.h"
using namespace std;

struct perf_stats{//data package for DIRECTLY output
	int n;//number of frame
	double acc_t;//accumulate time
	double avg_t;//average time
};

class logging{
protected:
	bool using_log;
	int indent_level;
	int n_update_weight, n_orient_all, n_propagation, n_halucinate;
	float t_update_weight, t_orient_all, t_propagation, t_halucinate;
	std::map<int, string*> Type_to_String;
	std::map<string, int> log_name_to_type;
	std::map<int, int> log_status;
	cudaEvent_t start, stop;

public:
	enum STATS{UPDATE_WEIGHT, ORIENT_ALL, PROPAGATION};
	enum LOG{
		PROCESS, // the test process
		INIT, // the init process, only once per test
		SIZE, // the size init, or resize
		ADD_SENSOR, // add sensor
		SAVING, // saving data
		LOADING, // loading data
		UP, // doing up
		MALLOC, // malloc memory
		REMALLOC, // remalloc, for expansion
		INIT_DATA, // init the data
		COPY_DATA, // copy the data
		AMPER, // amper
		DELAY, // delay
		AMPERAND, // amperand
		GENERATE_DELAY // generate delay
	};
	perf_stats STAT_UPDATE_WEIGHT, STAT_ORIENT_ALL, STAT_PROPAGATION;
	string str_init, str_size, str_add_sensor, str_up_GPU, str_saving, str_loading;
	string str_delay, str_generate_delay, str_amper, str_amperand;
	string str_malloc, str_remalloc, str_copy_data, str_init_data;
	string str_num_sim;
	string str_process;
	string agent_name;
	std::map<int, string> log_section_name;
	static long long GPU_MEM, CPU_MEM;
	static void add_GPU_MEM(int mem);
	static void add_CPU_MEM(int mem);
	static int num_sim;

public:
	logging(string name, string filter);
	~logging();

	void reset_all();
	void createEvent();
	void record_start();
	void record_stop(int LOG_TYPE);
	void finalize_stats(perf_stats &stats, int n, double acc_t);
	void finalize_log();
	void append_log(int LOG_TYPE, string info);
	void append_process(int LOG_FROM, int LOG_TO);
	void add_indent();
	void reduce_indent();
	void init_Type_to_String();
	void init_log_name_to_type();
	void init_log_section_name();
	void read_log_filter(string filter);
};

#endif