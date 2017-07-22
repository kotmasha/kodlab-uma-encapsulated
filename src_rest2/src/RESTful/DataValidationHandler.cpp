#include "DataValidationHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

DataValidationHandler::DataValidationHandler(logManager *log_access):AdminHandler(log_access) {
	_log_access = log_access;
	UMA_INITIAL_SENSOR_SIZE = U("initial_sensor_size");
}

void DataValidationHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
	json::value data = request.extract_json().get();
	if (paths[0] == UMA_SNAPSHOT) {
		validate_snapshot(world, data, request);
		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
	json::value message;
	message[MESSAGE] = json::value::string(U("cannot handle ") + paths[0] + U(" object"));
	request.reply(status_codes::BadRequest, message);
}

void DataValidationHandler::handle_update(World *world, vector<string_t> &paths, http_request &request) {

}

void DataValidationHandler::handle_read(World *world, vector<string_t> &paths, http_request &request) {

}

void DataValidationHandler::handle_delete(World *world, vector<string_t> &paths, http_request &request) {

}

void DataValidationHandler::validate_snapshot(World *world, json::value &data, http_request &request) {
	string agent_id, snapshot_id;
	int base_sensor_size;
	try {
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
		base_sensor_size = get_int_input(data, UMA_INITIAL_SENSOR_SIZE, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}
	
	Agent *agent = NULL;
	Snapshot *snapshot = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	if (!get_snapshot_by_id(agent, snapshot_id, snapshot, request)) return;
	bool status = snapshot->validate(base_sensor_size);
	if (status) {
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		json::value message;
		message[MESSAGE] = json::value::string(U("Snapshot validation succeed"));
		request.reply(status_codes::Created, message);
	}
	else {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[MESSAGE] = json::value::string(U("Cannot validate the snapshot"));
		request.reply(status_codes::BadRequest, message);
	}
}

DataValidationHandler::~DataValidationHandler(){}