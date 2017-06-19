#include "World.h"
#include "Agent.h"
#include "logging.h"
#include "logManager.h"

World::World(){
	_rmdir("log");
	_mkdir("log");
	_log_path = "log";
	_log = new logManager(logging::VERBOSE, _log_path, "world.txt", typeid(*this).name());
	_log->info() << "A new world is created";
}

bool World::add_agent(string name, string uuid){
	if (_agents.find(uuid) != _agents.end()) {
		_log->error() << "Cannot create a duplicate agent " + uuid;
		return false;
	}
	_agents[uuid] = new Agent(name, uuid);
	_log->info() << "An agent " + uuid + "(" + name + ") is created";
	return true;
}

Agent *World::getAgent(const string agent_id) {
	if (_agents.find(agent_id) != _agents.end()) {
		_log->debug() << "Agent " + agent_id + " is found";
		return _agents[agent_id];
	}
	_log->warn() << "No agent " + agent_id + " is found";
	return NULL;
}

World::~World(){}