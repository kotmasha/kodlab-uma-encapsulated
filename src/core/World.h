#ifndef _WORLD_
#define _WORLD_

#include "Global.h"
using namespace std;

class Agent;

/*
This is the world class, it maintains all the agent object
*/
class DLL_PUBLIC World{
private:
	static World *_world;
	std::map<string, Agent*> _agents, _load_agents;

public:
	static std::map<string, std::map<string, string>> core_info;

public:
	World();
	static World *instance();
	Agent *add_agent(const string &agent_id, int type = AGENT_TYPE::STATIONARY);
	Agent *getAgent(const string &agent_id);
	void delete_agent(const string &agent_id);
	const vector<vector<string>> getAgentInfo();
	//void save_world(string &name);
	//void load_world(string &name);
	//void merge_test();
	~World();
};

#endif