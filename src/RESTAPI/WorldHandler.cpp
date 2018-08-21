#include "WorldHandler.h"
#include "World.h"
#include "Agent.h"
#include "UMAException.h"

static Logger serverLogger("Server", "log/UMA_server.log");
WorldHandler::WorldHandler(const string &handler_name): UMARestHandler(handler_name) {}

void WorldHandler::handleCreate(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle POST " + request.get_request_url(), false, &serverLogger);
}

void WorldHandler::handleUpdate(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle PUT " + request.get_request_url(), false, &serverLogger);
}

void WorldHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/world") {
		getWorld(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void WorldHandler::handleDelete(UMARestRequest &request) {
	throw UMABadOperationException("Cannot handle DELETE " + request.get_request_url(), false, &serverLogger);
}

void WorldHandler::getWorld(UMARestRequest &request) {
	vector<string> experimentIds = World::instance()->getExperimentInfo();
	request.set_message("Get world info");
	request.set_data("experiment_ids", experimentIds);
}