#include "AgentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"
#include "Logger.h"

static Logger serverLogger("Server", "log/UMA_server.log");

AgentHandler::AgentHandler(const string &handlerName): UMARestHandler(handlerName) {}

void AgentHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/agent") {
		createAgent(request);
		return;
	}

	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void AgentHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle PUT " + requestUrl, false, &serverLogger);
}

void AgentHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/agent") {
		getAgent(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void AgentHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/agent") {
		deleteAgent(request);
		return;
	}

	throw UMABadOperationException("Cannot handle DELETE" + requestUrl, false, &serverLogger);
}

void AgentHandler::createAgent(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string agentType = request.getStringData("type");
	UMA_AGENT type;
	if (agentType == "default") type = UMA_AGENT::AGENT_STATIONARY;
	else type = UMA_AGENT::AGENT_QUALITATIVE;
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->createAgent(agentId, type);

	request.setMessage("Agent=" + agentId + " is created");
}

void AgentHandler::getAgent(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	vector<vector<string>> snapshotIds = agent->getSnapshotInfo();

	request.setMessage("Get agent info");
	request.setData("snapshot_ids", snapshotIds);
}

void AgentHandler::deleteAgent(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->deleteAgent(agentId);

	request.setMessage("Agent=" + agentId + " is deleted");
}

AgentHandler::~AgentHandler() {}
