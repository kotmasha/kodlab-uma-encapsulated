#include "ExperimentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"
#include "Logger.h"

static Logger serverLogger("Server", "log/UMA_server.log");
ExperimentHandler::ExperimentHandler(const string &handlerName) : UMARestHandler(handlerName) {}

void ExperimentHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/experiment") {
		createExperiment(request);
		return;
	}
	else if (requestUrl == "/UMA/object/experiment/save") {
		saveExperiment(request);
		return;
	}
	else if (requestUrl == "/UMA/object/experiment/load") {
		loadExperiment(request);
		return;
	}

	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void ExperimentHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle PUT " + requestUrl, false, &serverLogger);
}

void ExperimentHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/experiment") {
		getExperiment(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void ExperimentHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/experiment") {
		deleteExperiment(request);
		return;
	}

	throw UMABadOperationException("Cannot handle DELETE" + requestUrl, false, &serverLogger);
}

void ExperimentHandler::createExperiment(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	World::instance()->createExperiment(experimentId);

	request.setMessage("Experiment=" + experimentId + " is created");
}

void ExperimentHandler::getExperiment(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	vector<vector<string>> agentIds = experiment->getAgentInfo();

	request.setMessage("Get experiment info");
	request.setData("agent_ids", agentIds);
}

void ExperimentHandler::deleteExperiment(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	World::instance()->deleteExperiment(experimentId);

	request.setMessage("Experiment=" + experimentId + " is deleted");
}

void ExperimentHandler::saveExperiment(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->saveExperiment();
}

void ExperimentHandler::loadExperiment(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");

	Experiment::loadExperiment(experimentId);
}

ExperimentHandler::~ExperimentHandler() {}
