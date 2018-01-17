#include "Agent.h"
#include "Snapshot.h"
#include "Logger.h"
#include "UMAException.h"

static Logger agentLogger("Agent", "log/agent.log");

/*
Agent::Agent(ifstream &file) {
	int uuid_length = -1;
	file.read((char *)(&uuid_length), sizeof(int));
	_uuid = string(uuid_length, ' ');
	file.read(&_uuid[0], uuid_length * sizeof(char));

	_log_dir = "log/Agent_" + _uuid;
	_log = new logManager(logging::VERBOSE, _log_dir, "agent.txt", typeid(*this).name());

	int snapshot_size = -1;
	file.read((char *)(&snapshot_size), sizeof(int));
	
	for (int i = 0; i < snapshot_size; ++i) {
		string log_dir = _log_dir + "/Snapshot_";
		Snapshot_Stationary *snapshot = new Snapshot_Stationary(file, log_dir);
		_snapshots[snapshot->_uuid] = snapshot;
	}

	_log->info() << "An agent " + _uuid + " is loaded";
}
*/

Agent::Agent(const string &uuid, const string &dependency): _uuid(uuid), _dependency(dependency + ":" + _uuid){
	_t = 0;

	agentLogger.info("An agent " + uuid + " is created", _dependency);
}

void Agent::add_snapshot_stationary(const string &uuid){
	if (_snapshots.find(uuid) != _snapshots.end()) {
		agentLogger.error("Cannot create a duplicate snapshot!", _dependency);
		throw UMAException("Cannot create a duplicate snapshot!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::DUPLICATE);
	}
	_snapshots[uuid] = new Snapshot_Stationary(uuid, _dependency);
	agentLogger.info("A Snapshot Stationary " + uuid + " is created", _dependency);
}

Snapshot *Agent::getSnapshot(const string &snapshot_id){
	if (_snapshots.find(snapshot_id) != _snapshots.end()) {
		return _snapshots[snapshot_id];
	}
	agentLogger.warn("No snapshot " + snapshot_id + " is found", _dependency);
	throw UMAException("Cannot find the snapshot id!", UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
}

/*
void Agent::save_agent(ofstream &file) {
	//write uuid
	int uuid_length = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuid_length), sizeof(int));
	file.write(_uuid.c_str(), uuid_length * sizeof(char));
	int snapshot_size = _snapshots.size();
	file.write(reinterpret_cast<const char *>(&snapshot_size), sizeof(int));
	for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
		it->second->save_snapshot(file);
	}
}

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

const vector<string> Agent::getSnapshotInfo() const{
	vector<string> results;
	for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
		results.push_back(it->first);
	}
	return results;
}

void Agent::delete_snapshot(const string &snapshot_id) {
	if (_snapshots.find(snapshot_id) == _snapshots.end()) {
		throw UMAException("Cannot find the agent to delete " + snapshot_id, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::NO_RECORD);
	}
	delete _snapshots[snapshot_id];
	_snapshots[snapshot_id] = NULL;
	_snapshots.erase(snapshot_id);
	agentLogger.info("Snapshot deleted", _dependency);
}

Agent::~Agent(){
	try {
		for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
			delete it->second;
			_snapshots[it->first] = NULL;
		}
	}
	catch (exception &e) {
		agentLogger.error("Fatal error while trying to delete agent: " + _uuid, _dependency);
		throw UMAException("Fatal error in Agent destruction function", UMAException::ERROR_LEVEL::FATAL, UMAException::SERVER);
	}
	agentLogger.info("Deleted the agent " + _uuid);
}