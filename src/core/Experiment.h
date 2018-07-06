#ifndef _EXPERIMENT_
#define _EXPERIMENT_

#include "Global.h"

class Agent;

class DLL_PUBLIC Experiment {
public:
	Experiment(const string &name, const string &dependency="");
	~Experiment();
	Agent *createAgent(const string &agentId, int type = AGENT_TYPE::STATIONARY);
	Agent *getAgent(const string &agentId);
	void deleteAgent(const string &agentId);
	const vector<vector<string>> getAgentInfo();

protected:
	const string _name;
	const string _dependency;
	std::map<string, Agent*> _agents;
};

#endif