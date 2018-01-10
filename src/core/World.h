#ifndef _WORLD_
#define _WORLD_

#include "Global.h"
using namespace std;

class Agent;

/*
This is the world class, it maintains all the agent object
*/
class DLL_PUBLIC World{
public:
	static std::map<string, std::map<string, string>> core_info;

public:
	World();
	static void add_agent(const string &agent_id);
	static void delete_agent(const string &agent_id);
	static Agent *getAgent(const string &agent_id);
	static const vector<string> getAgentInfo();
	//void save_world(string &name);
	//void load_world(string &name);
	//void merge_test();
	~World();
};

#endif