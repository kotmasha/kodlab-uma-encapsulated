#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"
#include "logging.h"
#include "UMAException.h"
#include "DataManager.h"

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

Agent::Agent(string uuid){
	_uuid = uuid;
	_log_dir = "log/Agent_" + uuid;
	_log = new logManager(logging::VERBOSE, _log_dir, "agent.txt", typeid(*this).name());
	_log->info() << "An agent " + uuid + " is created";
}

void Agent::add_snapshot_stationary(string &uuid){
	if (_snapshots.find(uuid) != _snapshots.end()) {
		_log->error() << "Cannot create a duplicate snapshot!";
		throw CoreException("Cannot create a duplicate snapshot!", CoreException::ERROR, status_codes::Conflict);
	}
	string log_dir = _log_dir + "/Snapshot_" + uuid;
	_snapshots[uuid] = new Snapshot_Stationary(uuid, log_dir);
	_log->info() << "A Snapshot Stationary " + uuid + " is created";
}

Snapshot *Agent::getSnapshot(string &snapshot_id) {
	if (_snapshots.find(snapshot_id) != _snapshots.end()) {
		return _snapshots[snapshot_id];
	}
	_log->warn() << "No snapshot " + snapshot_id + " is found";
	throw CoreException("Cannot find the snapshot id!", CoreException::ERROR, status_codes::NotFound);
}

vector<float> Agent::decide(vector<bool> &obs_plus, vector<bool> &obs_minus, double phi, bool active) {
	vector<float> result;
	result.push_back(_snapshots["plus"]->decide(obs_plus, phi, active));
	result.push_back(_snapshots["minus"]->decide(obs_minus, phi, !active));
	return result;
}

vector<vector<bool>> Agent::getCurrent() {
	vector<vector<bool>> result;
	result.push_back(_snapshots["plus"]->getDM()->getCurrent());
	result.push_back(_snapshots["minus"]->getDM()->getCurrent());
	return result;
}

vector<vector<bool>> Agent::getPrediction() {
	vector<vector<bool>> result;
	result.push_back(_snapshots["plus"]->getDM()->getPrediction());
	result.push_back(_snapshots["minus"]->getDM()->getPrediction());
	return result;
}

vector<vector<bool>> Agent::getTarget() {
	vector<vector<bool>> result;
	result.push_back(_snapshots["plus"]->getDM()->getTarget());
	result.push_back(_snapshots["minus"]->getDM()->getTarget());
	return result;
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
vector<string> Agent::getSnapshotInfo() {
	vector<string> results;
	for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
		results.push_back(it->first);
	}
	return results;
}

void Agent::delete_snapshot(string &snapshot_id) {
	if (_snapshots.find(snapshot_id) == _snapshots.end()) {
		throw CoreException("Cannot find the agent to delete " + snapshot_id, CoreException::ERROR, status_codes::NotFound);
	}
	delete _snapshots[snapshot_id];
	_snapshots[snapshot_id] = NULL;
	_snapshots.erase(snapshot_id);
	_log->info() << "Snapshot deleted";
}

Agent::~Agent(){
	try {
		for (auto it = _snapshots.begin(); it != _snapshots.end(); ++it) {
			delete it->second;
			_snapshots[it->first] = NULL;
		}
	}
	catch (exception &e) {
		_log->error() << "Fatal error while trying to delete agent: " + _uuid;
		throw CoreException("Fatal error in Agent destruction function", CoreException::FATAL, status_codes::ServiceUnavailable);
	}
	_log->info() << "Deleted the agent " + _uuid;
}