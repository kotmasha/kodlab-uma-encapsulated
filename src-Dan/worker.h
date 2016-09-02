#ifndef _WORKER_
#define _WORKER_

#include <string>
using namespace std;

class worker{
public:
	worker();
	worker(string sensor_name1,string sensor_name2,int sensor_id1,int sensor_id2);
	static void add_time();
	static void reset_time();

public:
	string sensor_name1,sensor_name2;
	int sensor_id1,sensor_id2;
	double *wij,*w_ij,*wi_j,*w_i_j;//4 addresses pointing to weights
	bool *dij,*d_ij,*di_j,*d_i_j;
	bool *dji,*d_ji,*dj_i,*d_j_i;
	double threshold;
	double epsilon;
	double q;
	static int t;
};

#endif