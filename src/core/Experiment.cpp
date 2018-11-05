#include "Experiment.h"
#include "Agent.h"
#include "World.h"
#include "Logger.h"
#include "UMAException.h"
#include "PropertyMap.h"

static Logger experimentLogger("Experiment", "log/experiment.log");

Experiment::Experiment(const string &uuid) : UMACoreObject(uuid, UMA_OBJECT::EXPERIMENT, World::instance()) {
	experimentLogger.info("An experiment is created, experimentId=" + uuid, this->getParentChain());
}

void Experiment::addAgent(Agent * const agent) {
	string agentId = agent->getUUID();
	if (_agents.end() != _agents.find(agentId)) {
		throw UMADuplicationException("Cannot add a duplicate agent, agentId=" + agentId, false, &experimentLogger, this->getParentChain());
	}
	
	_agents[agentId] = agent;
	experimentLogger.info("An agent is added, agentId=" + agentId, this->getParentChain());
}

Agent *Experiment::createAgent(const string &agentId, UMA_AGENT type, PropertyMap *ppm) {
	if (_agents.find(agentId) != _agents.end()) {
		throw UMADuplicationException("Cannot create a duplicate agent, agentId=" + agentId, false, &experimentLogger, this->getParentChain());
	}
	switch (type) {
	case UMA_AGENT::AGENT_STATIONARY:
		_agents[agentId] = new Agent(agentId, this, UMA_AGENT::AGENT_STATIONARY, ppm); break;
	case UMA_AGENT::AGENT_QUALITATIVE:
		_agents[agentId] = new AgentQualitative(agentId, this, ppm); break;
	case UMA_AGENT::AGENT_DISCOUNTED:
		_agents[agentId] = new AgentDiscounted(agentId, this, ppm); break;
	case UMA_AGENT::AGENT_EMPIRICAL:
		_agents[agentId] = new AgentEmpirical(agentId, this, ppm); break;
	default:
		throw UMAInvalidArgsException("The input agent type is invalid, type=" + getUMAAgentName(type));
	}
	experimentLogger.info("An agent is created, agentId=" + agentId + ", type=" + getUMAAgentName(type), this->getParentChain());
	return _agents[agentId];
}

Agent *Experiment::getAgent(const string &agentId) {
	if (_agents.find(agentId) != _agents.end()) {
		return _agents[agentId];
	}
	throw UMANoResourceException("Cannot find object, agentId=" + agentId, false, &experimentLogger, this->getParentChain());
}

void Experiment::deleteAgent(const string &agentId) {
	if (_agents.find(agentId) == _agents.end()) {
		throw UMANoResourceException("Cannot find object, agentId=" + agentId, false, &experimentLogger, this->getParentChain());
	}
	delete _agents[agentId];
	_agents[agentId] = nullptr;
	_agents.erase(agentId);
	experimentLogger.info("Agent is deleted, agentId=" + agentId, this->getParentChain());
}

void Experiment::saveExperiment() {
	ofstream file;
	file.open(_uuid + ".uma", ios::out | ios::binary);
	int agentSize = _agents.size();
	file.write(reinterpret_cast<const char *>(&agentSize), sizeof(int));
	for (auto it = _agents.begin(); it != _agents.end(); ++it) {
		it->second->saveAgent(file);
	}
	file.close();
	experimentLogger.info("experiment_id=" + _uuid + " is saved to " + _uuid, this->getParentChain());
}

Experiment *Experiment::loadExperiment(const string &experimentId) {
	ifstream file;
	Experiment *experiment = nullptr;
	try {
		file.open(experimentId + ".uma", ios::binary | ios::in);
		int agentSize = -1;
		file.read((char *)(&agentSize), sizeof(int));

		experiment = new Experiment(experimentId);
		experimentLogger.debug("will load " + to_string(agentSize) + " agents", experiment->getParentChain());

		for (int i = 0; i < agentSize; ++i) {
			Agent *agent = Agent::loadAgent(file, experiment);
			experiment->addAgent(agent);
		}

	}
	catch (exception &ex) {
		throw UMAInternalException("Cannot load the experiment=" + experimentId + " error=" + ex.what(), true, &experimentLogger);
	}
	file.close();
	experimentLogger.info("experiment=" + experimentId + " is successfully loaded!");

	return experiment;
}

/*
This function is trying to merge another experiment info with the current experiment, mainly used in test reloading
For agent that exist in exp, but not in current experiment, it will be ignored
For agent that exist in exp, and also in current experiment, an agent level merging is required.
Input: Experiment * exp, the experiment to be merged, remove indicate whether remove the exp after merging, default is true
*/
void Experiment::mergeExperiment(Experiment * const exp) {
	for (auto it = exp->_agents.begin(); it != exp->_agents.end(); ++it) {
		if (_agents.end() != _agents.find(it->first)) {
			// find the agent with the same name, will merge agent
			_agents[it->first]->mergeAgent(it->second);
		}
		else {
			experimentLogger.debug("agentId=" + it->first + " is not in the new experiment, will be ignored", this->getParentChain());
		}
	}

	World::instance()->deleteExperiment(exp->_uuid);
	experimentLogger.info("experimentId=" + exp->_uuid + " is successfully merged into experimentId=" + _uuid);
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
		experimentLogger.info("Experiment is deleted, experimentId=" + _uuid, this->getParentChain());
	}
	catch (exception &ex) {
		throw UMAInternalException("Fatal error deleting experiment, experimentId=" + _uuid, true, &experimentLogger, this->getParentChain());
	}
}
