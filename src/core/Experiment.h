#ifndef _EXPERIMENT_
#define _EXPERIMENT_

#include "Global.h"
#include "UMACoreObject.h"

class Agent;
class PropertyMap;

class DLL_PUBLIC Experiment: public UMACoreObject{
public:
	Experiment(const string &uuid);
	~Experiment();
	void addAgent(Agent * const agent);
	Agent *createAgent(const string &agentId, UMA_AGENT type, PropertyMap *ppm=nullptr);
	Agent *getAgent(const string &agentId);
	void deleteAgent(const string &agentId);
	void saveExperiment();
	static Experiment *loadExperiment(const string &experimentId);
	const vector<vector<string>> getAgentInfo();
	//void mergeExperiment(Experiment * const exp);

	friend class ExperimentSavingLoading;
	friend class UMASavingLoading;

protected:
	//the experiment's agents
	std::map<string, Agent*> _agents;
};

#endif