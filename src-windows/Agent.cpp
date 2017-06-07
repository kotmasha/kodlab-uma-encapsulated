#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"
#include "logging.h"

Agent::Agent(string name){
	_name = name;
	_log_path = "log/Agent_" + name;
	_log = new logManager(logging::VERBOSE, _log_path, "agent.txt", typeid(*this).name());
	_log->info() << "An Agent " + name + " is created";
}

Snapshot_Stationary Agent::add_snapshot_stationary(int base_sensor_size, double threshold, string name, vector<string> sensor_ids, vector<string> sensor_names, double q, bool cal_target){
	_snapshots[name] = new Snapshot_Stationary(base_sensor_size, threshold, name, sensor_ids, sensor_names, q, cal_target, _log_path + "/Snapshot_" + name);
	_log->info() << "A Snapshot Stationary " + name + " is created";
	return *(_snapshots[name]);
}


Agent::~Agent(){}