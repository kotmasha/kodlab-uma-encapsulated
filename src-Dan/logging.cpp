#include "logging.h"

logging::logging(bool using_log){
	this->using_log=using_log;
	reset_all();
}

logging::~logging(){}

void logging::reset_all(){
	t_halucinate=0;
	t_orient_all=0;
	t_propagation=0;
	t_update_weight=0;
	n_halucinate=0;
	n_orient_all=0;
	n_propagation=0;
	n_update_weight=0;

	str_init="";

	createEvent();
}

void logging::finalize_stats(perf_stats &stats,int n,double acc_t){
	stats.n=n;
	stats.acc_t=acc_t;
	stats.avg_t=acc_t/n;
}

void logging::finalize_log(){
	finalize_stats(STAT_UPDATE_WEIGHT,n_update_weight,t_update_weight);
	finalize_stats(STAT_ORIENT_ALL,n_orient_all,t_orient_all);
	finalize_stats(STAT_PROPAGATION,n_propagation,t_propagation);
}

void logging::append_log(int LOG_TYPE,string info){
	switch(LOG_TYPE){
	case LOG::INIT:
		str_init+=info;
		break;
	}
}