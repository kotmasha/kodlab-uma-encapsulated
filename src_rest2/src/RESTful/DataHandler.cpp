#include "DataHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "logManager.h"

DataHandler::DataHandler(string handler_factory):AdminHandler(handler_factory) {
	UMA_TARGET = U("target");
	UMA_CURRENT = U("current");
	UMA_PREDICTION = U("prediction");

	UMA_TARGET_LIST = U("target_list");
	UMA_SIGNALS = U("signals");
	UMA_OBSERVE = U("observe");

	UMA_DATA_SIZE = U("data_size");

	UMA_WEIGHTS = U("weights");
	UMA_DIRS = U("dirs");
	UMA_THRESHOLDS = U("thresholds");
}

void DataHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void DataHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());

	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	DataManager *dm = snapshot->getDM();

	if (path == U("/UMA/data/observe") && check_field(data, UMA_OBSERVE, false)) {
		vector<bool> signals = get_bool1d_input(data, UMA_OBSERVE);
		dm->setObserve(signals);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("observe value set"));
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/current") && check_field(data, UMA_CURRENT, false)) {
		vector<bool> signals = get_bool1d_input(data, UMA_CURRENT);
		dm->setCurrent(signals);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Customized current value set"));
		response.set_body(message);
		return;
	}

	throw ClientException("The coming put request has nothing to update", ClientException::CLIENT_ERROR, status_codes::NotAcceptable);
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
		vector<json::value> json_weights;
		vector_double2d_to_array(weights, json_weights);
		json::value return_data = json::value::array(json_weights);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get weights value"));
		message[DATA] = json::value();
		message[DATA][U("weights")] = return_data;
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/dirs")) {
		vector<vector<bool> > dirs = dm->getDir2D();
		vector<json::value> json_dirs;
		vector_bool2d_to_array(dirs, json_dirs);
		json::value return_data = json::value::array(json_dirs);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get dirs value"));
		message[DATA] = json::value();
		message[DATA][U("dirs")] = return_data;
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/thresholds")) {
		vector<vector<double> > thresholds = dm->getThreshold2D();
		vector<json::value> json_thresholds;
		vector_double2d_to_array(thresholds, json_thresholds);
		json::value return_data = json::value::array(json_thresholds);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get thresholds value"));
		message[DATA] = json::value();
		message[DATA][U("thresholds")] = return_data;
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/negligible")) {
		vector<bool> negligible = dm->getNegligible();
		vector<json::value> json_negligible;
		vector_bool_to_array(negligible, json_negligible);
		json::value return_data = json::value::array(json_negligible);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get negligible value"));
		message[DATA] = json::value();
		message[DATA][U("negligible")] = return_data;
		response.set_body(message);
		return;
	}
	else if (path == U("/UMA/data/data_size")) {
		std::map<string, int> size_info = dm->getSizeInfo();
		json::value json_size_info = convert_size_info(size_info);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("get size info"));
		message[DATA] = json::value();
		message[DATA][U("sizes")] = json_size_info;
		response.set_body(message);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}


json::value DataHandler::convert_size_info(const std::map<string, int> &size_info) {
	json::value size;
	for (auto it = size_info.begin(); it != size_info.end(); ++it) {
		string s = it->first;
		size[string_to_string_t(s)] = json::value(it->second);
	}
	return size;
}

void DataHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {

}

DataHandler::~DataHandler() {}