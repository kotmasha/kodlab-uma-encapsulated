#include "WorldHandler.h"
#include "World.h"
#include "Agent.h"
#include "UMAException.h"
#include "Logger.h"

static Logger serverLogger("Server", "log/UMA_server.log");
WorldHandler::WorldHandler(const string &handlerName): UMARestHandler(handlerName) {}

void WorldHandler::handleCreate(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle POST " + request.getRequestUrl(), false, &serverLogger);
}

void WorldHandler::handleUpdate(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle PUT " + request.getRequestUrl(), false, &serverLogger);
}

void WorldHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/world") {
		getWorld(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void WorldHandler::handleDelete(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle DELETE " + request.getRequestUrl(), false, &serverLogger);
}

void WorldHandler::getWorld(UMARestRequest &request) {
	vector<string> experimentIds = World::instance()->getExperimentInfo();
	request.setMessage("Get world info");
	request.setData("experiment_ids", experimentIds);
}