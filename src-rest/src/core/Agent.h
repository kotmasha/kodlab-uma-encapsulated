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
	string _uuid;
	string _log_dir;
	logManager *_log;

public:
	Agent(string name, string uuid);
	bool add_snapshot_stationary(string name, string uuid);
	Snapshot *getSnapshot(string snapshot_id);
	~Agent();
};

#endif