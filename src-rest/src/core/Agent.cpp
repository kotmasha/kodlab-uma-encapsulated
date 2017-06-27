#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"
#include "logging.h"

Agent::Agent(ifstream &file) {
	int uuid_length = -1;
	file.read((char *)(&uuid_length), sizeof(int));
	_uuid = string(uuid_length, ' ');
	file.read(&_uuid[0], uuid_length * sizeof(char));
	_name = _uuid;

	_log_dir = "log/Agent_" + _uuid;
	_log = new logManager(logging::VERBOSE, _log_dir, "agent.txt", typeid(*this).name());

	int snapshot_size = -1;
	file.read((char *)(&snapshot_size), sizeof(int));
	
	for (int i = 0; i < snapshot_size; ++i) {
		Snapshot_Stationary *snapshot = new Snapshot_Stationary(file, _log_dir + "/Snapshot_");
		_snapshots[snapshot->_uuid] = snapshot;
	}

	_log->info() << "An agent " + _uuid + "(" + _name + ") is loaded";
}

Agent::Agent(string name, string uuid){
	_name = name;
	_uuid = uuid;
	_log_dir = "log/Agent_" + uuid;
	_log = new logManager(logging::VERBOSE, _log_dir, "agent.txt", typeid(*this).name());
	_log->info() << "An agent " + uuid + "(" + name + ") is created";
}

bool Agent::add_snapshot_stationary(string name, string uuid){
	if (_snapshots.find(uuid) != _snapshots.end()) {
		_log->error() << "Cannot create a duplicate snapshot!";
		return false;
	}
	_snapshots[uuid] = new Snapshot_Stationary(name, uuid, _log_dir + "/Snapshot_" + name);
	_log->info() << "A Snapshot Stationary " + uuid + "(" + name + ") is created";
	return true;
}

Snapshot *Agent::getSnapshot(string snapshot_id) {
	if (_snapshots.find(snapshot_id) != _snapshots.end()) {
		_log->debug() << "Snapshot " + snapshot_id + " is found";
		return _snapshots[snapshot_id];
	}
	_log->warn() << "No snapshot " + snapshot_id + " is found";
	return NULL;
}

vector<float> Agent::decide(vector<bool> &signal, double phi, bool active) {
	vector<float> result;
	result.push_back(_snapshots[_uuid + "_plus"]->decide(signal, phi, active));
	result.push_back(_snapshots[_uuid + "_minus"]->decide(signal, phi, !active));
	return result;
}

vector<vector<bool>> Agent::getCurrent() {
	vector<vector<bool>> result;
	result.push_back(_snapshots[_uuid + "_plus"]->getCurrent());
	result.push_back(_snapshots[_uuid + "_minus"]->getCurrent());
	return result;
}

vector<vector<bool>> Agent::getPrediction() {
	vector<vector<bool>> result;
	result.push_back(_snapshots[_uuid + "_plus"]->getPrediction());
	result.push_back(_snapshots[_uuid + "_minus"]->getPrediction());
	return result;
}

vector<vector<bool>> Agent::getTarget() {
	vector<vector<bool>> result;
	result.push_back(_snapshots[_uuid + "_plus"]->getTarget());
	result.push_back(_snapshots[_uuid + "_minus"]->getTarget());
	return result;
}

void Agent::setName(string name) {
	_name = name;
	_log->info() << "agent name changed to " + name;
}

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

Agent::~Agent(){}