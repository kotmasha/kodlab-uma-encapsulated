#include "WorldHandler.h"
#include "World.h"
#include "Agent.h"

WorldHandler::WorldHandler(string handler_factory) :AdminHandler(handler_factory) {}

void WorldHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void WorldHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void WorldHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/world")) {
		get_world(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void WorldHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void WorldHandler::get_world(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	vector<string> agent_ids = world->getAgentInfo();

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Get world info"));
	message[DATA] = json::value();
	vector<json::value> json_agent_ids;
	vector_string_to_array(agent_ids, json_agent_ids);
	message[DATA][U("agent_ids")] = json::value::array(json_agent_ids);
	message[DATA][U("agent_count")] = json::value(agent_ids.size());
	response.set_body(message);
}