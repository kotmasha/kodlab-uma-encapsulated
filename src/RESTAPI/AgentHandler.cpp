#include "AgentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"
#include "Logger.h"
#include "PropertyMap.h"

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
	//bad temp solution
	double q = -1;
	double threshold = -1;
	if (request.checkDataField("q")) {
		q = request.getDoubleData("q");
	}
	if (request.checkDataField("threshold")) {
		threshold = request.getDoubleData("threshold");
	}
	PropertyMap *ppm = new PropertyMap();
	if (q >= 0) ppm->add("q", to_string(q));
	if (threshold >= 0) ppm->add("threshold", to_string(threshold));

	UMA_AGENT type;
	if (agentType == "default") type = UMA_AGENT::AGENT_STATIONARY;
	else if(agentType == "qualitative") type = UMA_AGENT::AGENT_QUALITATIVE;
	else if (agentType == "discounted") type = UMA_AGENT::AGENT_DISCOUNTED;
	else type = UMA_AGENT::AGENT_EMPIRICAL;
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->createAgent(agentId, type, ppm);

	request.setMessage("Agent=" + agentId + " is created");

	delete ppm;
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
