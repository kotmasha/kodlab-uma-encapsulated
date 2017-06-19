#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"
#include "logging.h"

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


Agent::~Agent(){}