#include "Experiment.h"
#include "Agent.h"
#include "Logger.h"
#include "UMAException.h"

static Logger experimentLogger("Experiment", "log/experiment.log");

Experiment::Experiment(const string &name, const string &dependency) : _name(name), _dependency(dependency) {

}

Agent *Experiment::createAgent(const string &agent_id, int type) {
	if (_agents.find(agent_id) != _agents.end()) {
		experimentLogger.error("Cannot create a duplicate agent " + agent_id);
		throw UMAException("Cannot create a duplicate agent " + agent_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::DUPLICATE);
	}
	if (AGENT_TYPE::STATIONARY == type) {
		_agents[agent_id] = new Agent(agent_id, _dependency + "::" + _name);
	}
	else {
		_agents[agent_id] = new Agent_qualitative(agent_id, _dependency + "::" + _name);
	}

	experimentLogger.info("An agent " + agent_id + " is created, with the type " + to_string(type));
	return _agents[agent_id];
}

Agent *Experiment::getAgent(const string &agent_id) {
	if (_agents.find(agent_id) != _agents.end()) {
		return _agents[agent_id];
	}
	experimentLogger.warn("No agent " + agent_id + " is found");
	throw UMAException("Cannot find the agent id!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
}

void Experiment::deleteAgent(const string &agent_id) {
	if (_agents.find(agent_id) == _agents.end()) {
		throw UMAException("Cannot find the agent to delete " + agent_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	delete _agents[agent_id];
	_agents[agent_id] = nullptr;
	_agents.erase(agent_id);
	experimentLogger.info("Agent deleted");
}

const vector<vector<string>> Experiment::getAgentInfo() {
	vector<vector<string>> results;
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		vector<string> tmp;
		tmp.push_back(it->first);

		const int type = it->second->getType();
		string s_type = "";
		if (AGENT_TYPE::STATIONARY == type) s_type = "default";
		else if (AGENT_TYPE::QUALITATIVE == type) s_type = "qualitative";
		tmp.push_back(s_type);

		results.push_back(tmp);
	}
	return results;
}

Experiment::~Experiment() {
	try {
		for (auto it = _agents.begin(); it != _agents.end(); ++it) {
			delete it->second;
		}
		experimentLogger.info("experiment=" + _name + " has been successfully deleted");
	}
	catch (exception &ex) {
		experimentLogger.error("Fatal error while trying to delete experiment=" + _name);
		throw UMAException("Fatal error while trying to delete experiment=" + _name, UMAException::ERROR_LEVEL::FATAL, UMAException::ERROR_TYPE::SERVER);
	}
}