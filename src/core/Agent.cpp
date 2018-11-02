#include "Agent.h"
#include "World.h"
#include "Snapshot.h"
#include "Logger.h"
#include "UMAException.h"
#include "CoreService.h"
#include "PropertyMap.h"
#include "PropertyPage.h"

static Logger agentLogger("Agent", "log/agent.log");

Agent::Agent(const string &uuid, UMACoreObject *parent, UMA_AGENT type, PropertyMap *ppm): UMACoreObject(uuid, UMA_OBJECT::AGENT, parent), _type(type) {
	_t = 0;
	layerInConf();
	_ppm->extend(ppm);

	_pruningInterval = stoi(_ppm->get("pruning_interval"));
	_enableEnrichment = stoi(_ppm->get("enable_enrichment"));

	agentLogger.info("An agent is created, agentId=" + uuid + ", type=" + getUMAAgentName(type), this->getParentChain());
}

void Agent::layerInConf() {
	string confName = "Agent::" + UMACoreConstant::getUMAAgentName(_type);
	PropertyMap *pm = CoreService::instance()->getPropertyMap(confName);
	if (pm) {
		_ppm->extend(pm);
	}
}

void Agent::addSnapshot(Snapshot * const snapshot) {
	const string uuid = snapshot->getUUID();
	if (_snapshots.find(uuid) != _snapshots.end()) {
		throw UMADuplicationException("Cannot add a duplicate snapshot, snapshotId=" + uuid, false, &agentLogger, this->getParentChain());
	}

	_snapshots[uuid] = snapshot;
	agentLogger.info("A Snapshot is added, snapshotId=" + uuid, this->getParentChain());
}

Snapshot *Agent::createSnapshot(const string &uuid) {
	if (_snapshots.find(uuid) != _snapshots.end()) {
		throw UMADuplicationException("Cannot create a duplicate snapshot, snapshotId=" + uuid, false, &agentLogger, this->getParentChain());
	}
	_snapshots[uuid] = new Snapshot(uuid, this, getUMASnapshotTypeByAgent(_type));
	agentLogger.info("A Snapshot is created, snapshotId=" + uuid, this->getParentChain());
	return _snapshots[uuid];
}

Snapshot *Agent::getSnapshot(const string &snapshot_id){
	if (_snapshots.find(snapshot_id) != _snapshots.end()) {
		return _snapshots[snapshot_id];
	}
	throw UMANoResourceException("Cannot find the snapshot id!", false, &agentLogger, this->getParentChain());
}


void Agent::saveAgent(ofstream &file) {
	//write uuid
	int uuidLength = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuidLength), sizeof(int));
	file.write(_uuid.c_str(), uuidLength * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_type), sizeof(int));
	int snapshotSize = _snapshots.size();
	file.write(reinterpret_cast<const char *>(&snapshotSize), sizeof(int));
	for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
		it->second->saveSnapshot(file);
	}
}

Agent *Agent::loadAgent(ifstream &file, UMACoreObject *parent) {
	int uuidLength = -1;
	file.read((char *)(&uuidLength), sizeof(int));

	string uuid = string(uuidLength, ' ');
	file.read(&uuid[0], uuidLength * sizeof(char));

	UMA_AGENT type = UMA_AGENT::AGENT_STATIONARY;
	file.read((char *)(&type), sizeof(int));

	Agent *agent = nullptr;
	switch (type) {
	case UMA_AGENT::AGENT_QUALITATIVE:
		agent = new AgentQualitative(uuid, parent); break;
	case UMA_AGENT::AGENT_DISCOUNTED:
		agent = new AgentDiscounted(uuid, parent); break;
	case UMA_AGENT::AGENT_EMPIRICAL:
		agent = new AgentEmpirical(uuid, parent); break;
	default:
		agent = new Agent(uuid, parent, type); break;
	}

	int snapshotSize = -1;
	file.read((char *)(&snapshotSize), sizeof(int));
	agentLogger.debug("will load " + to_string(snapshotSize) + " snapshots", agent->getParentChain());

	for (int i = 0; i < snapshotSize; ++i) {
		UMA_SNAPSHOT sType = UMACoreConstant::getUMASnapshotTypeByAgent(type);
		Snapshot *snapshot = Snapshot::loadSnapshot(file, agent, sType);
		agent->addSnapshot(snapshot);
	}

	agentLogger.info("agent=" + uuid + " is successfully loaded", agent->getParentChain());

	return agent;
}

/*
void Agent::copy_test_data(Agent *agent) {
	for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
		string snapshot_id = it->first;
		Snapshot *snapshot = it->second;
		if (agent->_snapshots.find(snapshot_id) != agent->_snapshots.end()) {
			//if find the 'same' snapshot
			Snapshot *c_snapshot = agent->_snapshots[snapshot_id];
			snapshot->copy_test_data(c_snapshot);
			_log->info() << "Snapshot(" + snapshot_id + ") data merged";
		}
		else {
			_log->info() << "Cannot find snapshot(" + snapshot_id + ") data in old test, this should be a new snapshot";
		}
	}
}
*/

const vector<vector<string>> Agent::getSnapshotInfo() const {
	vector<vector<string>> results;
	for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
		vector<string> tmp;
		tmp.push_back(it->first);

		UMA_SNAPSHOT type = it->second->getType();
		tmp.push_back(getUMASnapshotName(type));

		results.push_back(tmp);
	}
	return results;
}

void Agent::deleteSnapshot(const string &snapshotId) {
	if (_snapshots.find(snapshotId) == _snapshots.end()) {
		throw UMANoResourceException("Cannot find the snapshot, snapshotId=" + snapshotId, false, &agentLogger, this->getParentChain());
	}
	delete _snapshots[snapshotId];
	_snapshots[snapshotId] = NULL;
	_snapshots.erase(snapshotId);
	agentLogger.info("Snapshot is deleted, snapshotId=" + snapshotId, this->getParentChain());
}

void Agent::setT(int t) {
	_t = t;
}

void Agent::setEnableEnrichment(bool enableEnrichment) {
	_enableEnrichment = enableEnrichment;
}

void Agent::setPruningInterval(int pruningInterval) {
	_pruningInterval = pruningInterval;
}

const int &Agent::getT() const {
	return _t;
}

const UMA_AGENT &Agent::getType() const {
	return _type;
}

const bool &Agent::getEnableEnrichment() const {
	return _enableEnrichment;
}

const int &Agent::getPruningInterval() const {
	return _pruningInterval;
}

bool Agent::doPruning() {
	return _pruningInterval && _t % _pruningInterval == 0;
}

Agent::~Agent(){
	try {
		for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
			delete it->second;
			_snapshots[it->first] = NULL;
		}
	}
	catch (exception &e) {
		throw UMAInternalException("Fatal error in Agent destruction function, agentId=" + _uuid, true, &agentLogger, this->getParentChain());
	}
	agentLogger.info("Agent is deleted, agentId=" + _uuid, this->getParentChain());
}

/*
-----------------------AgentQualitative class-----------------------
*/
AgentQualitative::AgentQualitative(const string &uuid, UMACoreObject *parent, PropertyMap *ppm)
	: Agent(uuid, parent, UMA_AGENT::AGENT_QUALITATIVE, ppm) {}

AgentQualitative::~AgentQualitative() {}

Snapshot *AgentQualitative::createSnapshot(const string &uuid) {
	if (_snapshots.find(uuid) != _snapshots.end()) {
		throw UMADuplicationException("Cannot create a duplicate snapshot, snapshotId=" + uuid, false, &agentLogger, this->getParentChain());
	}
	_snapshots[uuid] = new SnapshotQualitative(uuid, this);
	agentLogger.info("A Snapshot is created, snapshotId=" + uuid + ", type=Qualitative", this->getParentChain());
	return _snapshots[uuid];
}

/*
-----------------------AgentQualitative class-----------------------
*/

/*
-----------------------AgentDiscounted class-----------------------
*/
AgentDiscounted::AgentDiscounted(const string &uuid, UMACoreObject *parent, PropertyMap *ppm)
	: Agent(uuid, parent, UMA_AGENT::AGENT_DISCOUNTED, ppm) {}

AgentDiscounted::~AgentDiscounted() {}

Snapshot *AgentDiscounted::createSnapshot(const string &uuid) {
	if (_snapshots.find(uuid) != _snapshots.end()) {
		throw UMADuplicationException("Cannot create a duplicate snapshot, snapshotId=" + uuid, false, &agentLogger, this->getParentChain());
	}
	_snapshots[uuid] = new SnapshotDiscounted(uuid, this);
	agentLogger.info("A Snapshot is created, snapshotId=" + uuid + ", type=Discounted", this->getParentChain());
	return _snapshots[uuid];
}

/*
-----------------------AgentDiscounted class-----------------------
*/

/*
-----------------------AgentEmpirical class-----------------------
*/
AgentEmpirical::AgentEmpirical(const string &uuid, UMACoreObject *parent, PropertyMap *ppm)
	: Agent(uuid, parent, UMA_AGENT::AGENT_EMPIRICAL, ppm) {}

AgentEmpirical::~AgentEmpirical() {}

Snapshot *AgentEmpirical::createSnapshot(const string &uuid) {
	if (_snapshots.find(uuid) != _snapshots.end()) {
		throw UMADuplicationException("Cannot create a duplicate snapshot, snapshotId=" + uuid, false, &agentLogger, this->getParentChain());
	}
	_snapshots[uuid] = new SnapshotEmpirical(uuid, this);
	agentLogger.info("A Snapshot is created, snapshotId=" + uuid + ", type=Empirical", this->getParentChain());
	return _snapshots[uuid];
}

/*
-----------------------AgentEmpirical class-----------------------
*/
