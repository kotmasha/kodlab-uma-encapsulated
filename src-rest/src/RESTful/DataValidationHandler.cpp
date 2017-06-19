#include "DataValidationHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

DataValidationHandler::DataValidationHandler(logManager *log_access):AdminHandler(log_access) {
	_log_access = log_access;
}

void DataValidationHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
	json::value &data = request.extract_json().get();
	if (paths[0] == UMA_SNAPSHOT) {
		validate_snapshot(world, data, request);
		return;
	}

	_log_access->error() << request.absolute_uri().to_string() + L" 400";
	request.reply(status_codes::BadRequest, json::value::string(L"cannot handle " + paths[0] + L" object"));
}

void DataValidationHandler::handle_update(World *world, vector<string_t> &paths, http_request &request) {

}

void DataValidationHandler::handle_read(World *world, vector<string_t> &paths, http_request &request) {

}

void DataValidationHandler::handle_delete(World *world, vector<string_t> &paths, http_request &request) {

}

void DataValidationHandler::validate_snapshot(World *world, json::value &data, http_request &request) {
	if (!check_field(data, UMA_AGENT_ID, request)) return;
	if (!check_field(data, UMA_SNAPSHOT_ID, request)) return;
	string_t agent_id, snapshot_id;
	try {
		agent_id = data[UMA_AGENT_ID].as_string();
		snapshot_id = data[UMA_SNAPSHOT_ID].as_string();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		return;
	}
	//convent from wstring to string
	std::string s_agent_id(agent_id.begin(), agent_id.end());
	std::string s_snapshot_id(snapshot_id.begin(), snapshot_id.end());
	
	Agent *agent = world->getAgent(s_agent_id);
	if (agent == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		request.reply(status_codes::NotFound, json::value::string(L"Cannot find the agent id!"));
		return;
	}
	Snapshot *snapshot = agent->getSnapshot(s_snapshot_id);
	if (snapshot == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		request.reply(status_codes::NotFound, json::value::string(L"Cannot find the snapshot id!"));
		return;
	}
	bool status = snapshot->validate();
	if (status) {
		_log_access->info() << request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, json::value::string(L"Snapshot Validation succeed"));
	}
	else {
		_log_access->error() << request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"Cannot validate the snapshot!"));
	}
}

DataValidationHandler::~DataValidationHandler(){}