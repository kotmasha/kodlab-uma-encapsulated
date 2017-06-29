#ifndef _WORLD_
#define _WORLD_

#include "Global.h"
using namespace std;

class Agent;
class logManager;

/*
This is the world class, it maintains all the agent object
*/
class World{
protected:
	std::map<string, Agent*> _agents;
	string _log_path;
	logManager *_log;
public:
	World();
	bool add_agent(string uuid);
	Agent *getAgent(const string agent_id);
	void save_world(string &name);
	void load_world(string &name);
	~World();
};

#endif