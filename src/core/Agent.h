#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
using namespace std;
class Snapshot;
class SnapshotQualitative;
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
	int _pruningInterval;
	bool _enableEnrichment;
	friend class World;

public:
	//Agent(ifstream &file);
	Agent(const string &uuid, const string &dependency, const int type = AGENT_TYPE::STATIONARY);
	virtual Snapshot *createSnapshot(const string &uuid);
	Snapshot *getSnapshot(const string &snapshotId);
	void deleteSnapshot(const string &snapshotId);
	//void save_agent(ofstream &file);
	//void copy_test_data(Agent *agent);
	const int &getT() const;
	const int &getPruningInterval() const;
	const bool &getEnableEnrichment() const;
	const int &getType() const;

	void setT(int t);
	void setEnableEnrichment(bool enableEnrichment);
	void setPruningInterval(int pruningInterval);
	bool doPruning();

	const vector<vector<string>> getSnapshotInfo() const;
	virtual ~Agent();
};

class DLL_PUBLIC AgentQualitative : public Agent {
public:
	AgentQualitative(const string &uuid, const string &dependency);
	~AgentQualitative();

	virtual Snapshot *createSnapshot(const string &uuid);
};

#endif