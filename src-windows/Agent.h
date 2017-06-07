#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
using namespace std;
class Snapshot;
class Snapshot_Stationary;
class Snapshot_Forgetful;
class logManager;

/*
This is the Agent class, store several snapshot, and makes decision based on the python world observation
*/
class Agent{
protected:
	//the snapshot map, from name to pointer
	std::map<string, Snapshot_Stationary*> _snapshots;
	//the sensor name
	string _name;
	string _log_path;
	logManager *_log;

public:
	Agent(string name);
	Snapshot_Stationary add_snapshot_stationary(int base_sensor_size, double threshold, string name, vector<string> sensor_ids, vector<string> sensor_names, double q, bool cal_target);
	~Agent();
};

#endif