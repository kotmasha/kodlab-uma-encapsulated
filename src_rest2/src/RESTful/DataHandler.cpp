#include "DataHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "logManager.h"

DataHandler::DataHandler(string handler_factory, logManager *log_access):AdminHandler(handler_factory, log_access) {
	UMA_TARGET = U("target");
	UMA_CURRENT = U("current");
	UMA_PREDICTION = U("prediction");

	UMA_TARGET_LIST = U("target_list");
	UMA_SIGNALS = U("signals");

	UMA_DATA_SIZE = U("data_size");

	UMA_WEIGHTS = U("weights");
	UMA_DIRS = U("dirs");
	UMA_THRESHOLDS = U("thresholds");
}

void DataHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void DataHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/data/signals")) {
		update_signals(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void DataHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();

	if (path == U("/UMA/data/current")) {
		vector<bool> current = dm->getCurrent();
		vector<json::value> json_current;
		vector_bool_to_array(current, json_current);
		json::value return_data = json::value::array(json_current);
		
		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get current value"));
		message[DATA] = json::value();
		message[DATA][U("current")] = return_data;
		response.set_body(message);
		return;
	}
	else if(path == U("/UMA/data/prediction")){
		vector<bool> prediction = dm->getPrediction();
		vector<json::value> json_prediction;
		vector_bool_to_array(prediction, json_prediction);
		json::value return_data = json::value::array(json_prediction);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get prediction value"));
		message[DATA] = json::value();
		message[DATA][U("current")] = return_data;
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/target")) {
		vector<bool> target = dm->getTarget();
		vector<json::value> json_target;
		vector_bool_to_array(target, json_target);
		json::value return_data = json::value::array(json_target);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get target value"));
		message[DATA] = json::value();
		message[DATA][U("current")] = return_data;
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/weights")) {
		vector<vector<double> > weights = dm->getWeight2D();
		vector<json::value> json_target;
		vector_double2d_to_array(weights, json_target);
		json::value return_data = json::value::array(json_target);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get target value"));
		message[DATA] = json::value();
		message[DATA][U("current")] = return_data;
		response.set_body(message);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
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
	DataManager *dm = snapshot->getDM();

	if (check_field(data, UMA_SIGNALS, false)) {
		vector<bool> signals = get_bool1d_input(data, UMA_SIGNALS);
		dm->setObserve(signals);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Customized signal set"));
		response.set_body(message);
		return;
	}

	throw ClientException("The coming put request has nothing to update", ClientException::ERROR, status_codes::NotAcceptable);
}

DataHandler::~DataHandler() {}