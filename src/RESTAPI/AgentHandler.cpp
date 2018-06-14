#include "AgentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"

AgentHandler::AgentHandler(const string &handler_name): UMARestHandler(handler_name) {}

void AgentHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/agent") {
		createAgent(request);
		return;
	}

	throw UMAException("Cannot handle POST " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	throw UMAException("Cannot handle PUT " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/agent") {
		getAgent(request);
		return;
	}

	throw UMAException("Cannot handle GET " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/agent") {
		deleteAgent(request);
		return;
	}

	throw UMAException("Cannot handle DELETE" + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::createAgent(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string agentType = request.get_string_data("type");
	int type = 0;
	if (agentType == "default") type = AGENT_TYPE::STATIONARY;
	else type = AGENT_TYPE::QUALITATIVE;
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->createAgent(agentId, type);

	request.set_message("Agent=" + agentId + " is created");
}

void AgentHandler::getAgent(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	vector<vector<string>> snapshotIds = agent->getSnapshotInfo();

	request.set_message("Get agent info");
	request.set_data("snapshot_ids", snapshotIds);
}

void AgentHandler::deleteAgent(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->deleteAgent(agentId);

	request.set_message("Agent=" + agentId + " is deleted");
}

AgentHandler::~AgentHandler() {}
