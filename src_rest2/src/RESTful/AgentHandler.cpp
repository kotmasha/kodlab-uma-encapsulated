#include "AgentHandler.h"
#include "World.h"
#include "Agent.h"
#include "logManager.h"
#include "UMAException.h"

AgentHandler::AgentHandler(string handler_factory, logManager *log_access):AdminHandler(handler_factory, log_access) {}

void AgentHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/agent")) {
		create_agent(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void AgentHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {}

void AgentHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/agent")) {
		get_agent(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void AgentHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/agent")) {
		delete_agent(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);

}

void AgentHandler::create_agent(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string uuid = get_string_input(data, UMA_AGENT_ID);

	world->add_agent(uuid);
	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Agent created"));
	response.set_body(message);
}

void AgentHandler::get_agent(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	Agent *agent = world->getAgent(agent_id);
	vector<string> snapshot_ids = agent->getSnapshotInfo();

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Get agent info"));
	message[DATA] = json::value();
	vector<json::value> json_snapshot_ids;
	vector_string_to_array(snapshot_ids, json_snapshot_ids);
	message[DATA][U("snapshot_ids")] = json::value::array(json_snapshot_ids);
	message[DATA][U("snapshot_count")] = json::value(snapshot_ids.size());
	response.set_body(message);
}

void AgentHandler::delete_agent(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);

	world->delete_agent(agent_id);
	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Agent deleted"));
	response.set_body(message);
}

AgentHandler::~AgentHandler() {}
