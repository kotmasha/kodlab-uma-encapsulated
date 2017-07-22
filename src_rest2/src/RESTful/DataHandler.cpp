#include "DataHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"

DataHandler::DataHandler(logManager *log_access):AdminHandler(log_access) {
	UMA_TARGET = U("target");
	UMA_CURRENT = U("current");
	UMA_PREDICTION = U("prediction");

	UMA_TARGET_LIST = U("target_list");
}

void DataHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
	json::value data = request.extract_json().get();

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
	json::value message;
	message[MESSAGE] = json::value::string(U("cannot handle ") + paths[0] + U(" object"));
	request.reply(status_codes::BadRequest, message);
}

void DataHandler::handle_update(World *world, vector<string_t> &paths, http_request &request) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	json::value data = request.extract_json().get();
	string agent_id, snapshot_id;
	vector<bool> target;
	try {
		agent_id = get_string_input(query, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID, request);
		target = get_bool1d_input(data, UMA_TARGET_LIST, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	Snapshot *snapshot = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	if (!get_snapshot_by_id(agent, snapshot_id, snapshot, request)) return;

	if (paths[0] == UMA_TARGET) {
		snapshot->setTarget(target);
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 201");
		json::value message;
		message[MESSAGE] = json::value::string(U("target value updated"));
		request.reply(status_codes::OK, message);
		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
	json::value message;
	message[MESSAGE] = json::value::string(U("cannot handle ") + paths[0] + U(" object"));
	request.reply(status_codes::BadRequest, message);
}

void DataHandler::handle_read(World *world, vector<string_t> &paths, http_request &request) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id, snapshot_id;
	try {
		agent_id = get_string_input(query, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	Snapshot *snapshot = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	if (!get_snapshot_by_id(agent, snapshot_id, snapshot, request)) return;

	if (paths[0] == UMA_CURRENT) {
		vector<bool> current = snapshot->getCurrent();
		vector<json::value> json_current;
		vector_bool_to_array(current, json_current);
		json::value return_data = json::value::array(json_current);
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 200");

		json::value message;
		message[MESSAGE] = json::value::string(U("get current value"));
		message[U("data")] = return_data;
		request.reply(status_codes::OK, message);
		return;
	}
	else if(paths[0] == UMA_PREDICTION){
		vector<bool> prediction = snapshot->getPrediction();
		vector<json::value> json_prediction;
		vector_bool_to_array(prediction, json_prediction);
		json::value return_data = json::value::array(json_prediction);
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 200");

		json::value message;
		message[MESSAGE] = json::value::string(U("get prediction value"));
		message[U("data")] = return_data;
		request.reply(status_codes::OK, message);
		return;
	}
	else if (paths[0] == UMA_TARGET) {
		vector<bool> target = snapshot->getTarget();
		vector<json::value> json_target;
		vector_bool_to_array(target, json_target);
		json::value return_data = json::value::array(json_target);
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 200");

		json::value message;
		message[MESSAGE] = json::value::string(U("get target value"));
		message[U("data")] = return_data;
		request.reply(status_codes::OK, message);
		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
	json::value message;
	message[MESSAGE] = json::value::string(U("cannot handle ") + paths[0] + U(" object"));
	request.reply(status_codes::BadRequest, message);
}

void DataHandler::handle_delete(World *world, vector<string_t> &paths, http_request &request) {

}

DataHandler::~DataHandler() {}