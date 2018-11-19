#include "AgentHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "UMAException.h"
#include "Logger.h"
#include "PropertyMap.h"
#include "PropertyPage.h"
#include "CoreService.h"

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
	
	vector<string> keys = CoreService::instance()->getPropertyMap("Agent")->getKeys();
	PropertyMap ppm;
	request.getValueInKeys(keys, ppm);

	UMA_AGENT type;
	if (agentType == "default") type = UMA_AGENT::AGENT_STATIONARY;
	else if(agentType == "qualitative") type = UMA_AGENT::AGENT_QUALITATIVE;
	else if (agentType == "discounted") type = UMA_AGENT::AGENT_DISCOUNTED;
	else type = UMA_AGENT::AGENT_EMPIRICAL;
	Experiment *experiment = World::instance()->getExperiment(experimentId);
	experiment->createAgent(agentId, type, &ppm);

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

	request.setMessage("AgentId=" + agentId + " is deleted");
}

void AgentHandler::copyAgent(UMARestRequest &request) {
	const string experimentId1 = request.getStringData("experiment_id1");
	const string experimentId2 = request.getStringData("experiment_id2");
	const string agentId1 = request.getStringData("agent_id1");
	const string agentId2 = request.getStringData("agent_id2");

	Experiment *experiment1 = World::instance()->getExperiment(experimentId1);
	Agent *agent1 = experiment1->getAgent(agentId1);

	Experiment *experiment2 = World::instance()->getExperiment(experimentId2);
	UMA_AGENT type = agent1->getType();

	Agent *cAgent = nullptr;
	switch (type) {
	case UMA_AGENT::AGENT_STATIONARY: cAgent = new Agent(*agent1, experiment2, agentId2); break;
	case UMA_AGENT::AGENT_QUALITATIVE:
		cAgent = new AgentQualitative(*dynamic_cast<AgentQualitative*>(agent1), experiment2, agentId2); break;
	case UMA_AGENT::AGENT_EMPIRICAL:
		cAgent = new AgentEmpirical(*dynamic_cast<AgentEmpirical*>(agent1), experiment2, agentId2); break;
	case UMA_AGENT::AGENT_DISCOUNTED:
		cAgent = new AgentDiscounted(*dynamic_cast<AgentDiscounted*>(agent1), experiment2, agentId2); break;
	default:
		throw UMAInternalException("The agent type does not map to any existing type, experimentId="
			+ experiment1->getUUID() , true, &serverLogger);
	}

	experiment2->addAgent(cAgent);

	request.setMessage("AgentId=" + agentId1 + " is copied");
}

AgentHandler::~AgentHandler() {}
