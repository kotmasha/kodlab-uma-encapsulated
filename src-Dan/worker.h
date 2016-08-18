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
	string sensor_name1,sensor_name2;
	int sensor_id1,sensor_id2;
	double *ij,*_ij,*i_j,*_i_j;//4 addresses pointing to weights
	static int t;
};

#endif