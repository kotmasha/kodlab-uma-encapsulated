#include "WorldHandler.h"
#include "World.h"
#include "Agent.h"
#include "UMAException.h"

WorldHandler::WorldHandler(const string &handler_name): UMARestHandler(handler_name) {}

void WorldHandler::handleCreate(UMARestRequest &request) {
	throw UMAException("Cannot handle POST " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::handleUpdate(UMARestRequest &request) {
	throw UMAException("Cannot handle PUT " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/world") {
		getWorld(request);
		return;
	}

	throw UMAException("Cannot handle GET " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::handleDelete(UMARestRequest &request) {
	throw UMAException("Cannot handle DELETE " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::getWorld(UMARestRequest &request) {
	vector<string> experimentIds = World::instance()->getExperimentInfo();
	request.set_message("Get world info");
	request.set_data("experiment_ids", experimentIds);
}