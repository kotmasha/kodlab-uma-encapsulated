#include "AgentHandler.h"
#include "World.h"
#include "Agent.h"
#include "UMAException.h"

AgentHandler::AgentHandler(const string &handler_name): UMARestHandler(handler_name) {}

void AgentHandler::handle_create(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/agent") {
		create_agent(request);
		return;
	}

	throw UMAException("Cannot handle POST " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::handle_update(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	throw UMAException("Cannot handle PUT " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::handle_read(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/agent") {
		get_agent(request);
		return;
	}

	throw UMAException("Cannot handle GET " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::handle_delete(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/agent") {
		delete_agent(request);
		return;
	}

	throw UMAException("Cannot handle DELETE" + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AgentHandler::create_agent(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string agent_type = request.get_string_data("type");
	int type = 0;
	if (agent_type == "default") type = AGENT_TYPE::STATIONARY;
	else type = AGENT_TYPE::QUALITATIVE;
	World::add_agent(agent_id, type);

	request.set_message("Agent created");
}

void AgentHandler::get_agent(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	Agent *agent = World::getAgent(agent_id);
	vector<vector<string>> snapshot_ids = agent->getSnapshotInfo();

	request.set_message("Get agent info");
	request.set_data("snapshot_ids", snapshot_ids);
	request.set_data("snapshot_count", (int)snapshot_ids.size());
}

void AgentHandler::delete_agent(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	World::delete_agent(agent_id);

	request.set_message("Agent deleted");
}

AgentHandler::~AgentHandler() {}
