#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
using namespace std;
class Snapshot;
class Snapshot_Qualitative;
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
	//agent type
	const int _type;
	//pruning interval
	int _pruning_interval;
	friend class World;

public:
	//Agent(ifstream &file);
	Agent(const string &uuid, const string &dependency, const int type = AGENT_TYPE::STATIONARY);
	virtual Snapshot *add_snapshot(const string &uuid);
	Snapshot *getSnapshot(const string &snapshot_id);
	void delete_snapshot(const string &snapshot_id);
	//void save_agent(ofstream &file);
	//void copy_test_data(Agent *agent);
	const int &getT() const;
	const int &getPruningInterval() const;
	void setT(int t);
	const int &getType() const;
	bool do_pruning();

	const vector<vector<string>> getSnapshotInfo() const;
	virtual ~Agent();
};

class DLL_PUBLIC Agent_qualitative : public Agent {
public:
	Agent_qualitative(const string &uuid, const string &dependency);
	~Agent_qualitative();

	virtual Snapshot *add_snapshot(const string &uuid);
};

#endif