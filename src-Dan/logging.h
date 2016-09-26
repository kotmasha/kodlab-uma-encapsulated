#ifndef _LOGGING_
#define _LOGGING_

#include <string>
using namespace std;

struct perf_stats{//data package for DIRECTLY output
	int n;//number of frame
	double acc_t;//accumulate time
	double avg_t;//average time
};

class logging{
protected:
	bool using_log;
	int n_update_weight,n_orient_all,n_propagation,n_halucinate;
	float t_update_weight,t_orient_all,t_propagation,t_halucinate;

public:
	enum STATS{UPDATE_WEIGHT,ORIENT_ALL,PROPAGATION};
	enum LOG{INIT};
	perf_stats STAT_UPDATE_WEIGHT,STAT_ORIENT_ALL,STAT_PROPAGATION;
	string str_init;

public:
	logging(bool using_log);
	~logging();

	void reset_all();
	void createEvent();
	void record_start();
	void record_stop(int LOG_TYPE);
	void finalize_stats(perf_stats &stats,int n,double acc_t);
	void finalize_log();
	void append_log(int LOG_TYPE,string info);
};

#endif