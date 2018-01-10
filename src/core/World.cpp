#include "World.h"
#include "Agent.h"
#include "Logger.h"
#include "ConfReader.h"
#include "UMAException.h"

static std::map<string, Agent*> _agents, _load_agents;
static Logger worldLogger("World", "log/world.log");

std::map<string, std::map<string, string>> World::core_info = ConfReader::read_conf("core.ini");

World::World(){
	worldLogger.info("A new world is created");
}

void World::add_agent(const string &agent_id){
	if (_agents.find(agent_id) != _agents.end()) {
		worldLogger.error("Cannot create a duplicate agent " + agent_id);
		throw UMAException("Cannot create a duplicate agent " + agent_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::DUPLICATE);
	}
	_agents[agent_id] = new Agent(agent_id, "world");
	worldLogger.info("An agent " + agent_id + " is created");
}

Agent *World::getAgent(const string &agent_id) {
	if (_agents.find(agent_id) != _agents.end()) {
		return _agents[agent_id];
	}
	worldLogger.warn("No agent " + agent_id + " is found");
	throw UMAException("Cannot find the agent id!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
}

void World::delete_agent(const string &agent_id) {
	if (_agents.find(agent_id) == _agents.end()) {
		throw UMAException("Cannot find the agent to delete " + agent_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	delete _agents[agent_id];
	_agents[agent_id] = NULL;
	_agents.erase(agent_id);
	worldLogger.info("Agent deleted");
}

/*
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
*/

/*
void World::load_world(string &name) {
	ifstream file;
	file.open(name + ".uma", ios::binary | ios::in);
	int agent_size = -1;
	file.read((char *)(&agent_size), sizeof(int));
	for (int i = 0; i < agent_size; ++i) {
		//Agent *agent = new Agent(file);
		//_load_agents[agent->_uuid] = agent;
	}
	file.close();
}
*/

/*
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
*/

const vector<string> World::getAgentInfo() {
	vector<string> results;
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		results.push_back(it->first);
	}
	return results;
}

World::~World(){}