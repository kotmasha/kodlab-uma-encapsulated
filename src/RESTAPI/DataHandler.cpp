#include "DataHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "UMAException.h"

DataHandler::DataHandler(const string &handler_name): UMARestHandler(handler_name) {
}

void DataHandler::handle_create(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	throw UMAException("Cannot handle POST " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void DataHandler::handle_update(UMARestRequest &request) {
	const string request_url = request.get_request_url();

	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();

	if (request_url == "/UMA/data/observe" && request.check_data_field("observe")) {
		vector<bool> observe = request.get_bool1d_data("observe");
		dm->setObserve(observe);

		request.set_message("observe value set");
		return;
	}
	else if (request_url == "/UMA/data/current" && request.check_data_field("current")) {
		vector<bool> current = request.get_bool1d_data("current");
		dm->setCurrent(current);

		request.set_message("Customized current value set");
		return;
	}

	throw UMAException("The coming put request has nothing to update", UMAException::ERROR_LEVEL::WARN, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void DataHandler::handle_read(UMARestRequest &request) {
	const string request_url = request.get_request_url();

	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();

	if (request_url == "/UMA/data/current") {
		vector<bool> current = dm->getCurrent();
		request.set_message("get current value");
		request.set_data("current", current);
		return;
	}
	else if(request_url == "/UMA/data/prediction"){
		vector<bool> prediction = dm->getPrediction();
		request.set_message("get prediction value");
		request.set_data("prediction", prediction);
		return;
	}
	else if (request_url == "/UMA/data/target") {
		vector<bool> target = dm->getTarget();
		request.set_message("get target value");
		request.set_data("target", target);
		return;
	}
	else if (request_url == "/UMA/data/weights") {
		vector<vector<double> > weights = dm->getWeight2D();
		request.set_message("get weights value");
		request.set_data("weights", weights);
		return;
	}
	else if (request_url == "/UMA/data/dirs") {
		vector<vector<bool> > dirs = dm->getDir2D();
		request.set_message("get dirs value");
		request.set_data("dirs", dirs);
		return;
	}
	else if (request_url == "/UMA/data/thresholds") {
		vector<vector<double> > thresholds = dm->getThreshold2D();
		request.set_message("get thresholds value");
		request.set_data("thresholds", thresholds);
		return;
	}
	else if (request_url == "/UMA/data/negligible") {
		vector<bool> negligible = dm->getNegligible();
		request.set_message("get negligible value");
		request.set_data("negligible", negligible);
		return;
	}
	else if (request_url == "/UMA/data/data_size") {
		std::map<string, int> size_info = dm->getSizeInfo();
		request.set_message("get size info");
		request.set_data("sizes", size_info);
		return;
	}

	throw UMAException("The coming put request has nothing to get", UMAException::ERROR_LEVEL::WARN, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void DataHandler::handle_delete(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	throw UMAException("Cannot handle DELETE " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

DataHandler::~DataHandler() {}