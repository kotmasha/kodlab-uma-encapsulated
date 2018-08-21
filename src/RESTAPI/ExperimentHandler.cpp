#include "ExperimentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"

static Logger serverLogger("Server", "log/UMA_server.log");
ExperimentHandler::ExperimentHandler(const string &handler_name) : UMARestHandler(handler_name) {}

void ExperimentHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/experiment") {
		createExperiment(request);
		return;
	}

	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void ExperimentHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	throw UMABadOperationException("Cannot handle PUT " + requestUrl, false, &serverLogger);
}

void ExperimentHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/experiment") {
		getExperiment(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void ExperimentHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/experiment") {
		deleteExperiment(request);
		return;
	}

	throw UMABadOperationException("Cannot handle DELETE" + requestUrl, false, &serverLogger);
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
