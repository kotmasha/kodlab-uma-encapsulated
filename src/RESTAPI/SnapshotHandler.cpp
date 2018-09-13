#include "SnapshotHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "UMAutil.h"
#include "Logger.h"

static Logger serverLogger("Server", "log/UMA_server.log");
SnapshotHandler::SnapshotHandler(const string &handlerName) : UMARestHandler(handlerName) {
}

void SnapshotHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
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

	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void SnapshotHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/snapshot") {
		updateSnapshot(request);
		return;
	}

	throw UMABadOperationException("Cannot handle PUT " + requestUrl, false, &serverLogger);
}

void SnapshotHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/snapshot") {
		getSnapshot(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void SnapshotHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/snapshot") {
		deleteSnapshot(request);
		return;
	}

	throw UMABadOperationException("Cannot handle DELETE " + requestUrl, false, &serverLogger);
}

void SnapshotHandler::createSnapshot(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	agent->createSnapshot(snapshotId);

	request.setMessage("Snapshot=" + snapshotId + " is created");
}

void SnapshotHandler::createInit(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->setInitialSize();

	request.setMessage("Initial size set to sensor size");
}

void SnapshotHandler::createAmper(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<vector<bool> > amperLists = request.getBool2dData("amper_lists");
	const vector<vector<string> > uuidLists = request.getString2dData("uuid_lists");
	const vector<pair<string, string>> uuidPairs = StrUtil::string2dToString1dPair(uuidLists);

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->ampers(amperLists, uuidPairs);

	request.setMessage("Amper made succeed");
}

void SnapshotHandler::createDelay(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<vector<bool> > amperLists = request.getBool2dData("delay_lists");
	const vector<vector<string> > uuidLists = request.getString2dData("uuid_lists");
	const vector<pair<string, string>> uuidPairs = StrUtil::string2dToString1dPair(uuidLists);

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->delays(amperLists, uuidPairs);

	request.setMessage("Delay made succeed");
}

void SnapshotHandler::createPruning(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const vector<bool> signals = request.getBool1dData("signals");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->pruning(signals);
	request.setMessage("Pruning made succeed");
}

void SnapshotHandler::getSnapshot(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");

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

	request.setMessage("Get snapshot info");
	request.setData("sensors", sensorInfo);
	request.setData("total", total);
	request.setData("q", q);
	request.setData("threshold", threshold);
	request.setData("auto_target", autoTarget);
	request.setData("propagate_mask", propagateMask);
	request.setData("initial_size", initialSize);
}

void SnapshotHandler::deleteSnapshot(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	agent->deleteSnapshot(snapshotId);
	request.setMessage("Snapshot=" + snapshotId + " is deleted");
}

void SnapshotHandler::updateSnapshot(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);

	if (request.checkDataField("q")) {
		const double q = request.getDoubleData("q");
		snapshot->setQ(q);

		request.setMessage("Q updated");
		return;
	}
	else if (request.checkDataField("threshold")) {
		const double threshold = request.getDoubleData("threshold");
		snapshot->setThreshold(threshold);

		request.setMessage("Threshold updated");
		return;
	}
	else if (request.checkDataField("auto_target")) {
		const bool autoTarget = request.getBoolData("auto_target");
		snapshot->setAutoTarget(autoTarget);

		request.setMessage("Auto target updated");
		return;
	}
	else if (request.checkDataField("propagate_mask")) {
		const bool propagateMask = request.getBoolData("propagate_mask");
		snapshot->setPropagateMask(propagateMask);

		request.setMessage("Propagate mask updated");
		return;
	}
	else if (request.checkDataField("initial_size")) {
		int initialSize = request.getIntData("initial_size");
		snapshot->setInitialSize(initialSize);

		request.setMessage("initial size updated");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

SnapshotHandler::~SnapshotHandler() {}