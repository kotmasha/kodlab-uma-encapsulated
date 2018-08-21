#include "DataHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "UMAException.h"

static Logger serverLogger("Server", "log/UMA_server.log");
DataHandler::DataHandler(const string &handler_name): UMARestHandler(handler_name) {
}

void DataHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void DataHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();

	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();

	if (requestUrl == "/UMA/data/observe" && request.check_data_field("observe")) {
		vector<bool> observe = request.get_bool1d_data("observe");
		dm->setObserve(observe);

		request.set_message("observe value set");
		return;
	}
	else if (requestUrl == "/UMA/data/current" && request.check_data_field("current")) {
		vector<bool> current = request.get_bool1d_data("current");
		dm->setCurrent(current);

		request.set_message("Customized current value set");
		return;
	}
	else if (requestUrl == "/UMA/data/target" && request.check_data_field("target")) {
		vector<bool> target = request.get_bool1d_data("target");
		dm->setTarget(target);

		request.set_message("Customized target value set");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

void DataHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();

	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();

	if (requestUrl == "/UMA/data/current") {
		vector<bool> current = dm->getCurrent();
		request.set_message("get current value");
		request.set_data("current", current);
		return;
	}
	else if(requestUrl == "/UMA/data/prediction"){
		vector<bool> prediction = dm->getPrediction();
		request.set_message("get prediction value");
		request.set_data("prediction", prediction);
		return;
	}
	else if (requestUrl == "/UMA/data/target") {
		vector<bool> target = dm->getTarget();
		request.set_message("get target value");
		request.set_data("target", target);
		return;
	}
	else if (requestUrl == "/UMA/data/weights") {
		vector<vector<double> > weights = dm->getWeight2D();
		request.set_message("get weights value");
		request.set_data("weights", weights);
		return;
	}
	else if (requestUrl == "/UMA/data/dirs") {
		vector<vector<bool> > dirs = dm->getDir2D();
		request.set_message("get dirs value");
		request.set_data("dirs", dirs);
		return;
	}
	else if (requestUrl == "/UMA/data/thresholds") {
		vector<vector<double> > thresholds = dm->getThreshold2D();
		request.set_message("get thresholds value");
		request.set_data("thresholds", thresholds);
		return;
	}
	else if (requestUrl == "/UMA/data/negligible") {
		vector<bool> negligible = dm->getNegligible();
		request.set_message("get negligible value");
		request.set_data("negligible", negligible);
		return;
	}
	else if (requestUrl == "/UMA/data/dataSize") {
		std::map<string, int> sizeInfo = dm->getSizeInfo();
		std::map<string, int> convertedSizeInfo = dm->convertSizeInfo(sizeInfo);
		request.set_message("get size info");
		request.set_data("sizes", convertedSizeInfo);
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to get", false, &serverLogger);
}

void DataHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	throw UMABadOperationException("Cannot handle DELETE " + requestUrl, false, &serverLogger);
}

DataHandler::~DataHandler() {}