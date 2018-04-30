#include "SnapshotHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "UMAutil.h"

SnapshotHandler::SnapshotHandler(const string &handler_name) : UMARestHandler(handler_name) {
}

void SnapshotHandler::handle_create(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/snapshot") {
		create_snapshot(request);
		return;
	}
	else if (request_url == "/UMA/object/snapshot/init") {
		create_init(request);
		return;
	}
	else if (request_url == "/UMA/snapshot/amper") {
		create_amper(request);
		return;
	}
	else if (request_url == "/UMA/snapshot/delay") {
		create_delay(request);
		return;
	}
	else if (request_url == "/UMA/snapshot/pruning") {
		create_pruning(request);
		return;
	}


	throw UMAException("Cannot handle POST " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::handle_update(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/snapshot") {
		update_snapshot(request);
		return;
	}

	throw UMAException("Cannot handle PUT " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::handle_read(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/snapshot") {
		get_snapshot(request);
		return;
	}

	throw UMAException("Cannot handle GET " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::handle_delete(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/snapshot") {
		delete_snapshot(request);
		return;
	}

	throw UMAException("Cannot handle DELETE " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::create_snapshot(UMARestRequest &request) {
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string agent_id = request.get_string_data("agent_id");

	Agent *agent = World::getAgent(agent_id);
	agent->add_snapshot(snapshot_id);

	request.set_message("Snapshot created");
}

void SnapshotHandler::create_init(UMARestRequest &request) {
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string agent_id = request.get_string_data("agent_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->setInitialSize();

	request.set_message("Initial size set to sensor size");
}

void SnapshotHandler::create_amper(UMARestRequest &request) {
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string agent_id = request.get_string_data("agent_id");
	const vector<vector<bool> > amper_lists = request.get_bool2d_data("amper_lists");
	const vector<vector<string> > uuid_lists = request.get_string2d_data("uuid_lists");
	const vector<pair<string, string>> uuid_pairs = StrUtil::string2d_to_string1d_pair(uuid_lists);

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->ampers(amper_lists, uuid_pairs);

	request.set_message("Amper made succeed");
}

void SnapshotHandler::create_delay(UMARestRequest &request) {
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string agent_id = request.get_string_data("agent_id");
	const vector<vector<bool> > amper_lists = request.get_bool2d_data("delay_lists");
	const vector<vector<string> > uuid_lists = request.get_string2d_data("uuid_lists");
	const vector<pair<string, string>> uuid_pairs = StrUtil::string2d_to_string1d_pair(uuid_lists);

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->delays(amper_lists, uuid_pairs);

	request.set_message("Delay made succeed");
}

void SnapshotHandler::create_pruning(UMARestRequest &request) {
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string agent_id = request.get_string_data("agent_id");
	const vector<bool> signals = request.get_bool1d_data("signals");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->pruning(signals);
	request.set_message("Pruning made succeed");
}

void SnapshotHandler::get_snapshot(UMARestRequest &request) {
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string agent_id = request.get_string_query("agent_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);

	vector<vector<string> > sensor_info = snapshot->getSensorInfo();
	double total = snapshot->getTotal();
	double q = snapshot->getQ();
	double threshold = snapshot->getThreshold();
	bool auto_target = snapshot->getAutoTarget();
	bool propagate_mask = snapshot->getPropagateMask();
	int initial_size = snapshot->getInitialSize();

	request.set_message("Get snapshot info");
	request.set_data("sensors", sensor_info);
	request.set_data("total", total);
	request.set_data("q", q);
	request.set_data("threshold", threshold);
	request.set_data("auto_target", auto_target);
	request.set_data("propagate_mask", propagate_mask);
	request.set_data("initial_size", initial_size);
}

void SnapshotHandler::delete_snapshot(UMARestRequest &request) {
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string agent_id = request.get_string_data("agent_id");

	Agent *agent = World::getAgent(agent_id);
	agent->delete_snapshot(snapshot_id);
	request.set_message("Snapshot deleted");
}

void SnapshotHandler::update_snapshot(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);

	if (request.check_data_field("q")) {
		const double q = request.get_double_data("q");
		snapshot->setQ(q);

		request.set_message("Q updated");
		return;
	}
	else if (request.check_data_field("threshold")) {
		const double threshold = request.get_double_data("threshold");
		snapshot->setThreshold(threshold);

		request.set_message("Threshold updated");
		return;
	}
	else if (request.check_data_field("auto_target")) {
		const bool auto_target = request.get_bool_data("auto_target");
		snapshot->setAutoTarget(auto_target);

		request.set_message("Auto target updated");
		return;
	}
	else if (request.check_data_field("propagate_mask")) {
		const bool propagate_mask = request.get_bool_data("propagate_mask");
		snapshot->setPropagateMask(propagate_mask);

		request.set_message("Propagate mask updated");
		return;
	}
	else if (request.check_data_field("initial_size")) {
		int initial_size = request.get_int_data("initial_size");
		snapshot->setInitialSize(initial_size);

		request.set_message("initial size updated");
		return;
	}

	throw UMAException("The coming put request has nothing to update", UMAException::ERROR_LEVEL::WARN, UMAException::ERROR_TYPE::BAD_OPERATION);
}

SnapshotHandler::~SnapshotHandler() {}