#include "DataHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"

DataHandler::DataHandler(string handler_factory, logManager *log_access):AdminHandler(handler_factory, log_access) {
	UMA_TARGET = U("target");
	UMA_CURRENT = U("current");
	UMA_PREDICTION = U("prediction");

	UMA_TARGET_LIST = U("target_list");
	UMA_SIGNALS = U("signals");
}

void DataHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + U(" 400");
	json::value message;
	message[MESSAGE] = json::value::string(U("cannot handle ") + path);
	request.reply(status_codes::BadRequest, message);
}

void DataHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/data/signals")) {
		update_signals(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void DataHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	/*
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

	if (path == U("/data/current")) {
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
	else if(path == U("/data/prediction")){
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
	else if (path == U("/data/target")) {
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
	message[MESSAGE] = json::value::string(U("cannot handle ") + path);
	request.reply(status_codes::BadRequest, message);
	*/
}

void DataHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {

}

void DataHandler::update_signals(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());

	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);

	if (check_field(data, UMA_SIGNALS, false)) {
		vector<bool> signals = get_bool1d_input(data, UMA_SIGNALS);
		snapshot->setObserve(signals);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Customized signal set"));
		response.set_body(message);
		return;
	}

	throw ClientException("The coming put request has nothing to update", ClientException::ERROR, status_codes::NotAcceptable);
}

DataHandler::~DataHandler() {}