#include "World.h"
#include "Agent.h"
#include "logging.h"
#include "logManager.h"

World::World(){
	//_rmdir("log");
#if defined(_WIN64)
	_mkdir("log");
#else 
	mkdir("log", 0777);
#endif
	_log_path = "log";
	_log = new logManager(logging::VERBOSE, _log_path, "world.txt", typeid(*this).name());
	_log->info() << "A new world is created";
}

bool World::add_agent(string uuid){
	if (_agents.find(uuid) != _agents.end()) {
		_log->error() << "Cannot create a duplicate agent " + uuid;
		return false;
	}
	_agents[uuid] = new Agent(uuid);
	_log->info() << "An agent " + uuid + " is created";
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

void World::save_world(string &name) {
	ofstream file;
	file.open(name + ".uma", ios::out | ios::binary);
	int agent_size = _agents.size();
	file.write(reinterpret_cast<const char *>(&agent_size), sizeof(int));
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		it->second->save_agent(file);
	}
	file.close();
}

void World::load_world(string &name) {
	ifstream file;
	file.open(name + ".uma", ios::binary | ios::in);
	int agent_size = -1;
	file.read((char *)(&agent_size), sizeof(int));
	for (int i = 0; i < agent_size; ++i) {
		Agent *agent = new Agent(file);
		_load_agents[agent->_uuid] = agent;
	}
	file.close();
}

void World::merge_test() {
	_log->info() << "start merging old test agent data with new test";
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		string agent_id = it->first;
		Agent *agent = it->second;
		if (_load_agents.find(agent_id) != _load_agents.end()) {
			//if find the 'same' agent
			Agent *c_agent = _load_agents[agent_id];
			agent->copy_test_data(c_agent);
			_log->info() << "Agent(" + agent_id + ") data merged";
		}
		else {
			_log->info() << "Cannot find agent(" + agent_id + ") data in old test, this should be a new agent";
		}
	}
	_log->info() << "finish merging all data";
	//TBD delete load_agent data
}


World::~World(){}