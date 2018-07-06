#include "SnapshotHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "UMAutil.h"

SnapshotHandler::SnapshotHandler(const string &handler_name) : UMARestHandler(handler_name) {
}

void SnapshotHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/snapshot") {
		createSnapshot(request);
		return;
	}
	else if (requestUrl == "/UMA/object/snapshot/init") {
		createInit(request);
		return;
	}
	else if (requestUrl == "/UMA/object/snapshot/amper") {
		createAmper(request);
		return;
	}
	else if (requestUrl == "/UMA/object/snapshot/delay") {
		createDelay(request);
		return;
	}
	else if (requestUrl == "/UMA/object/snapshot/pruning") {
		createPruning(request);
		return;
	}


	throw UMAException("Cannot handle POST " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/snapshot") {
		updateSnapshot(request);
		return;
	}

	throw UMAException("Cannot handle PUT " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/snapshot") {
		getSnapshot(request);
		return;
	}

	throw UMAException("Cannot handle GET " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/snapshot") {
		deleteSnapshot(request);
		return;
	}

	throw UMAException("Cannot handle DELETE " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SnapshotHandler::createSnapshot(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	agent->createSnapshot(snapshotId);

	request.set_message("Snapshot=" + snapshotId + " is created");
}

void SnapshotHandler::createInit(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->setInitialSize();

	request.set_message("Initial size set to sensor size");
}

void SnapshotHandler::createAmper(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<vector<bool> > amperLists = request.get_bool2d_data("amper_lists");
	const vector<vector<string> > uuidLists = request.get_string2d_data("uuid_lists");
	const vector<pair<string, string>> uuidPairs = StrUtil::string2dToString1dPair(uuidLists);

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->ampers(amperLists, uuidPairs);

	request.set_message("Amper made succeed");
}

void SnapshotHandler::createDelay(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<vector<bool> > amperLists = request.get_bool2d_data("delay_lists");
	const vector<vector<string> > uuidLists = request.get_string2d_data("uuid_lists");
	const vector<pair<string, string>> uuidPairs = StrUtil::string2dToString1dPair(uuidLists);

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->delays(amperLists, uuidPairs);

	request.set_message("Delay made succeed");
}

void SnapshotHandler::createPruning(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const vector<bool> signals = request.get_bool1d_data("signals");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->pruning(signals);
	request.set_message("Pruning made succeed");
}

void SnapshotHandler::getSnapshot(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);

	vector<vector<string> > sensorInfo = snapshot->getSensorInfo();
	double total = snapshot->getTotal();
	double q = snapshot->getQ();
	double threshold = snapshot->getThreshold();
	bool autoTarget = snapshot->getAutoTarget();
	bool propagateMask = snapshot->getPropagateMask();
	int initialSize = snapshot->getInitialSize();

	request.set_message("Get snapshot info");
	request.set_data("sensors", sensorInfo);
	request.set_data("total", total);
	request.set_data("q", q);
	request.set_data("threshold", threshold);
	request.set_data("auto_target", autoTarget);
	request.set_data("propagate_mask", propagateMask);
	request.set_data("initial_size", initialSize);
}

void SnapshotHandler::deleteSnapshot(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	agent->deleteSnapshot(snapshotId);
	request.set_message("Snapshot=" + snapshotId + " is deleted");
}

void SnapshotHandler::updateSnapshot(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);

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
		const bool autoTarget = request.get_bool_data("auto_target");
		snapshot->setAutoTarget(autoTarget);

		request.set_message("Auto target updated");
		return;
	}
	else if (request.check_data_field("propagate_mask")) {
		const bool propagateMask = request.get_bool_data("propagate_mask");
		snapshot->setPropagateMask(propagateMask);

		request.set_message("Propagate mask updated");
		return;
	}
	else if (request.check_data_field("initial_size")) {
		int initialSize = request.get_int_data("initial_size");
		snapshot->setInitialSize(initialSize);

		request.set_message("initial size updated");
		return;
	}

	throw UMAException("The coming put request has nothing to update", UMAException::ERROR_LEVEL::WARN, UMAException::ERROR_TYPE::BAD_OPERATION);
}

SnapshotHandler::~SnapshotHandler() {}