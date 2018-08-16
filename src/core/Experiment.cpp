#include "Experiment.h"
#include "Agent.h"
#include "Logger.h"
#include "UMAException.h"

static Logger experimentLogger("Experiment", "log/experiment.log");

Experiment::Experiment(const string &name, const string &dependency) : _name(name), _dependency(dependency) {

}

Agent *Experiment::createAgent(const string &agentId, int type) {
	if (_agents.find(agentId) != _agents.end()) {
		experimentLogger.error("Cannot create a duplicate agent " + agentId);
		throw UMAException("Cannot create a duplicate agent " + agentId, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::DUPLICATE);
	}
	if (AGENT_TYPE::STATIONARY == type) {
		_agents[agentId] = new Agent(agentId, _dependency + "::" + _name);
	}
	else {
		_agents[agentId] = new AgentQualitative(agentId, _dependency + "::" + _name);
	}

	experimentLogger.info("An agent " + agentId + " is created, with the type " + to_string(type));
	return _agents[agentId];
}

Agent *Experiment::getAgent(const string &agentId) {
	if (_agents.find(agentId) != _agents.end()) {
		return _agents[agentId];
	}
	experimentLogger.warn("No agent " + agentId + " is found");
	throw UMAException("Cannot find the agent id!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
}

void Experiment::deleteAgent(const string &agentId) {
	if (_agents.find(agentId) == _agents.end()) {
		string s = "Cannot find the agent to delete " + agentId;
		experimentLogger.error(s);
		throw UMAException(s, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	delete _agents[agentId];
	_agents[agentId] = nullptr;
	_agents.erase(agentId);
	experimentLogger.info("Agent deleted");
}

const vector<vector<string>> Experiment::getAgentInfo() {
	vector<vector<string>> results;
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		vector<string> tmp;
		tmp.push_back(it->first);

		const int type = it->second->getType();
		string sType = "";
		if (AGENT_TYPE::STATIONARY == type) sType = "default";
		else if (AGENT_TYPE::QUALITATIVE == type) sType = "qualitative";
		tmp.push_back(sType);

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