#include "WorldHandler.h"
#include "World.h"
#include "Agent.h"
#include "UMAException.h"

WorldHandler::WorldHandler(const string &handler_name): UMARestHandler(handler_name) {}

void WorldHandler::handle_create(UMARestRequest &request) {
	throw UMAException("Cannot handle POST " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::handle_update(UMARestRequest &request) {
	throw UMAException("Cannot handle PUT " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::handle_read(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/world") {
		get_world(request);
		return;
	}

	throw UMAException("Cannot handle GET " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::handle_delete(UMARestRequest &request) {
	throw UMAException("Cannot handle DELETE " + request.get_request_url(), UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void WorldHandler::get_world(UMARestRequest &request) {
	vector<vector<string>> agent_ids = World::instance()->getAgentInfo();
	request.set_message("Get world info");
	request.set_data("agent_ids", agent_ids);
	request.set_data("agent_count", (int)agent_ids.size());
}