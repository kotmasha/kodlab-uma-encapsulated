#ifndef _EXPERIMENT_
#define _EXPERIMENT_

#include "Global.h"

class Agent;

class DLL_PUBLIC Experiment {
public:
	Experiment(const string &name, const string &dependency="");
	~Experiment();
	Agent *createAgent(const string &agent_id, int type = AGENT_TYPE::STATIONARY);
	Agent *getAgent(const string &agent_id);
	void deleteAgent(const string &agent_id);
	const vector<vector<string>> getAgentInfo();

protected:
	const string _name;
	const string _dependency;
	std::map<string, Agent*> _agents;
};

#endif