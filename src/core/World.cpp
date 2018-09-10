#include "World.h"
#include "Experiment.h"
#include "Logger.h"
#include "ConfReader.h"
#include "UMAException.h"

World *World::_world = NULL;
static Logger worldLogger("World", "log/world.log");

World *World::instance() {
	if (!_world) {
		_world = new World();
		worldLogger.info("A new world is created");
	}
	return _world;
}

World::World(): UMACoreObject("World", UMA_OBJECT::WORLD, nullptr) {}

Experiment *World::createExperiment(const string &experimentId) {
	if (_experiments.end() != _experiments.find(experimentId)) {
		throw UMADuplicationException("Object already exist, experimentId=" + experimentId, false, &worldLogger);
	}

	_experiments[experimentId] = new Experiment(experimentId);
	worldLogger.info("A new experiment=" + experimentId + " is created!");
	return _experiments[experimentId];
}

Experiment *World::getExperiment(const string &experimentId) {
	if (_experiments.end() == _experiments.find(experimentId)) {
		throw UMANoResourceException("Cannot find object, experimentId=" + experimentId, false, &worldLogger);
	}

	return _experiments[experimentId];
}

void World::deleteExperiment(const string &experimentId) {
	if (_experiments.end() == _experiments.find(experimentId)) {
		throw UMANoResourceException("Cannot find object, experimentId=" + experimentId, false, &worldLogger);
	}

	delete _experiments[experimentId];
	_experiments[experimentId] = nullptr;
	_experiments.erase(experimentId);
	worldLogger.info("Experiment is deleted, experimentId=" + experimentId);
}

vector<string> World::getExperimentInfo() {
	vector<string> results;
	for (auto it = _experiments.begin(); it != _experiments.end(); ++it) {
		results.push_back(it->first);
	}
	return results;
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

World::~World(){}