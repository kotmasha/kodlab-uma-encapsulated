#include "ExperimentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"

ExperimentHandler::ExperimentHandler(const string &handler_name) : UMARestHandler(handler_name) {}

void ExperimentHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/experiment") {
		createExperiment(request);
		return;
	}

	throw UMAException("Cannot handle POST " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void ExperimentHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	throw UMAException("Cannot handle PUT " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void ExperimentHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/experiment") {
		getExperiment(request);
		return;
	}

	throw UMAException("Cannot handle GET " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void ExperimentHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/experiment") {
		deleteExperiment(request);
		return;
	}

	throw UMAException("Cannot handle DELETE" + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void ExperimentHandler::createExperiment(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	World::instance()->createExperiment(experimentId);

	request.set_message("Experiment=" + experimentId + " is created");
}

void ExperimentHandler::getExperiment(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	vector<vector<string>> agentIds = experiment->getAgentInfo();

	request.set_message("Get experiment info");
	request.set_data("agent_ids", agentIds);
}

void ExperimentHandler::deleteExperiment(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	World::instance()->deleteExperiment(experimentId);

	request.set_message("Experiment=" + experimentId + " is deleted");
}

ExperimentHandler::~ExperimentHandler() {}
