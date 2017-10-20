#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
using namespace std;
class Snapshot;
class Snapshot_Stationary;
class Snapshot_Forgetful;
class logManager;
class World;

/*
This is the Agent class, store several snapshot, and makes decision based on the python world observation
*/
class Agent{
protected:
	//the snapshot map, from name to pointer
	std::map<string, Snapshot_Stationary*> _snapshots;
	string _uuid;
	string _log_dir;
	logManager *_log;
	friend class World;

public:
	//Agent(ifstream &file);
	Agent(string uuid);
	void add_snapshot_stationary(string &uuid);
	Snapshot *getSnapshot(string &snapshot_id);
	void delete_snapshot(string &snapshot_id);
	vector<float> decide(vector<bool> &obs_plus, vector<bool> &obs_minus, double phi, bool active);
	vector<vector<bool>> getCurrent();
	vector<vector<bool>> getPrediction();
	vector<vector<bool>> getTarget();
	//void save_agent(ofstream &file);
	//void copy_test_data(Agent *agent);

	vector<string> getSnapshotInfo();
	~Agent();
};

#endif