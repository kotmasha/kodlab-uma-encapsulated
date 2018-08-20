#include "Experiment.h"
#include "Agent.h"
#include "World.h"
#include "Logger.h"
#include "UMAException.h"

static Logger experimentLogger("Experiment", "log/experiment.log");

Experiment::Experiment(const string &uuid) : UMACoreObject(uuid, UMA_OBJECT::EXPERIMENT, World::instance()) {
	experimentLogger.info("An experiment is created, experimentId=" + uuid);
}

Agent *Experiment::createAgent(const string &agentId, UMA_AGENT type) {
	if (_agents.find(agentId) != _agents.end()) {
		throw UMAException("Cannot create a duplicate agent, agentId=" + agentId, &experimentLogger);
	}
	switch (type) {
	case UMA_AGENT::AGENT_STATIONARY:
		_agents[agentId] = new Agent(agentId, this, UMA_AGENT::AGENT_STATIONARY); break;
	case UMA_AGENT::AGENT_QUALITATIVE:
		_agents[agentId] = new AgentQualitative(agentId, this); break;
	default:
		throw UMAInvalidArgsException("The input agent type is invalid, type=" + to_string(type));
	}
	experimentLogger.info("An agent is created, agentId=" + agentId + ", type=" + getUMAAgentName(type));
	return _agents[agentId];
}

Agent *Experiment::getAgent(const string &agentId) {
	if (_agents.find(agentId) != _agents.end()) {
		return _agents[agentId];
	}
	throw UMANoResourceException("Cannot find object, agentId=" + agentId, &experimentLogger);
}

void Experiment::deleteAgent(const string &agentId) {
	if (_agents.find(agentId) == _agents.end()) {
		throw UMAException("Cannot find object, agentId=" + agentId, &experimentLogger);
	}
	delete _agents[agentId];
	_agents[agentId] = nullptr;
	_agents.erase(agentId);
	experimentLogger.info("Agent is deleted, agentId=" + agentId);
}

const vector<vector<string>> Experiment::getAgentInfo() {
	vector<vector<string>> results;
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		vector<string> tmp;
		tmp.push_back(it->first);

		UMA_AGENT type = it->second->getType();
		tmp.push_back(getUMAAgentName(type));

		results.push_back(tmp);
	}
	return results;
}

Experiment::~Experiment() {
	try {
		for (auto it = _agents.begin(); it != _agents.end(); ++it) {
			delete it->second;
		}
		experimentLogger.info("Experiment is deleted, experimentId=" + _uuid);
	}
	catch (exception &ex) {
		throw UMAInternalException("Fatal error deleting experiment, experimentId=" + _uuid, &experimentLogger);
	}
}
