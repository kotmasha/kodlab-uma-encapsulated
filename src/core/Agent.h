#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
using namespace std;
class Snapshot;
class Snapshot_Stationary;
class Snapshot_Forgetful;
class World;

/*
This is the Agent class, store several snapshot, and makes decision based on the python world observation
*/
class DLL_PUBLIC Agent{
protected:
	//the snapshot map, from name to pointer
	std::map<const string, Snapshot*> _snapshots;
	//the agent unique id
	const string _uuid;
	//the agent dependency chain
	const string _dependency;
	//the iteration times
	int _t;
	friend class World;

public:
	//Agent(ifstream &file);
	Agent(const string &uuid, const string &dependency);
	void add_snapshot_stationary(const string &uuid);
	Snapshot *getSnapshot(const string &snapshot_id);
	void delete_snapshot(const string &snapshot_id);
	//void save_agent(ofstream &file);
	//void copy_test_data(Agent *agent);
	const int &getT() const;
	void setT(int t);

	const vector<string> getSnapshotInfo() const;
	virtual ~Agent();
};

#endif