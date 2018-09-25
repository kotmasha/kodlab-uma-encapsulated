#ifndef _AGENT_
#define _AGENT_

#include "Global.h"
#include "UMACoreObject.h"
using namespace std;
class Snapshot;
class SnapshotQualitative;
class World;
class PropertyMap;

/*
This is the Agent class, store several snapshot, and makes decision based on the python world observation
*/
class DLL_PUBLIC Agent: public UMACoreObject{
protected:
	//the snapshot map, from name to pointer
	std::map<const string, Snapshot*> _snapshots;
	//the iteration times
	int _t;
	//agent type
	UMA_AGENT _type;
	//pruning interval
	int _pruningInterval;
	//enable enrichment
	bool _enableEnrichment;
	friend class World;

public:
	//Agent(ifstream &file);
	Agent(const string &uuid, UMACoreObject *parent, UMA_AGENT type = UMA_AGENT::AGENT_STATIONARY, PropertyMap *ppm = nullptr);
	virtual Snapshot *createSnapshot(const string &uuid);
	Snapshot *getSnapshot(const string &snapshotId);
	void deleteSnapshot(const string &snapshotId);
	//void save_agent(ofstream &file);
	//void copy_test_data(Agent *agent);
	const int &getT() const;
	const int &getPruningInterval() const;
	const bool &getEnableEnrichment() const;
	const UMA_AGENT &getType() const;

	void setT(int t);
	void setEnableEnrichment(bool enableEnrichment);
	void setPruningInterval(int pruningInterval);
	bool doPruning();

	const vector<vector<string>> getSnapshotInfo() const;
	virtual ~Agent();

protected:
	void layerInConf();
};

class DLL_PUBLIC AgentQualitative : public Agent {
public:
	AgentQualitative(const string &uuid, UMACoreObject *parent, PropertyMap *ppm = nullptr);
	~AgentQualitative();

	virtual Snapshot *createSnapshot(const string &uuid);
};

class DLL_PUBLIC AgentDiscounted : public Agent {
public:
	AgentDiscounted(const string &uuid, UMACoreObject *parent, PropertyMap *ppm = nullptr);
	~AgentDiscounted();

	virtual Snapshot *createSnapshot(const string &uuid);
};

class DLL_PUBLIC AgentEmpirical : public Agent {
public:
	AgentEmpirical(const string &uuid, UMACoreObject *parent, PropertyMap *ppm = nullptr);
	~AgentEmpirical();

	virtual Snapshot *createSnapshot(const string &uuid);
};

#endif