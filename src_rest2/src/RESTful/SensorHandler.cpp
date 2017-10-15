#include "SensorHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "UMAException.h"

SensorHandler::SensorHandler(string handler_factory, logManager *log_access) :AdminHandler(handler_factory, log_access) {
	UMA_THRESHOLD = U("threshold");
	UMA_Q = U("q");
	UMA_AUTO_TARGET = U("auto_target");
	UMA_C_SID = U("c_sid");
	UMA_AMPER_LIST = U("amper_list");
	UMA_SENSOR1 = U("sensor1");
	UMA_SENSOR2 = U("sensor2");

	UMA_W = U("w");
	UMA_D = U("d");
	UMA_DIAG = U("diag");
}

void SensorHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/sensor")) {
		create_sensor(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SensorHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SensorHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	if (path == U("/UMA/object/sensor")) {
		get_sensor(world, request, response);
		return;
	}
	else if (path == U("/UMA/object/sensor_pair")) {
		get_sensor_pair(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SensorHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	if (path == U("/UMA/object/sensor")) {
		delete_sensor(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SensorHandler::get_sensor(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string sensor_id = get_string_input(query, UMA_SENSOR_ID);
	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	Sensor *sensor = snapshot->getSensor(sensor_id);

	int idx = sensor->getIdx();

	vector<int> amper_list_idx = sensor->getAmperList();
	vector<json::value> json_amper_list_idx;
	vector_int_to_array(amper_list_idx, json_amper_list_idx);

	vector<bool> amper_list = snapshot->getAmperList(sensor_id);
	vector<json::value> json_amper_list;
	vector_bool_to_array(amper_list, json_amper_list);

	vector<string> amper_list_ids = snapshot->getAmperListID(sensor_id);
	vector<json::value> json_amper_list_ids;
	vector_string_to_array(amper_list_ids, json_amper_list_ids);

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Sensor info get"));
	message[DATA] = json::value();
	message[DATA][U("amper_list_idx")] = json::value::array(json_amper_list_idx);
	message[DATA][U("amper_list")] = json::value::array(json_amper_list);
	message[DATA][U("amper_list_ids")] = json::value::array(json_amper_list_ids);
	message[DATA][U("idx")] = json::value(idx);
	response.set_body(message);
}

void SensorHandler::get_sensor_pair(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string sensor_id1 = get_string_input(query, UMA_SENSOR1);
	string sensor_id2 = get_string_input(query, UMA_SENSOR2);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	Sensor *sensor1 = snapshot->getSensor(sensor_id1);
	Sensor *sensor2 = snapshot->getSensor(sensor_id2);
	SensorPair *sensor_pair = snapshot->getSensorPair(sensor1, sensor2);
	double threshold = sensor_pair->getThreshold();

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Sensor Pair info get"));
	message[DATA] = json::value();
	message[DATA][U("threshold")] = json::value(threshold);
	response.set_body(message);
}

void SensorHandler::create_sensor(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	string sensor_id = get_string_input(data, UMA_SENSOR_ID);
	string c_sid = get_string_input(data, UMA_C_SID);
	vector<vector<double> > w = get_double2d_input(data, UMA_W);
	vector<vector<bool> > d = get_bool2d_input(data, UMA_D);
	vector<double> diag = get_double1d_input(data, UMA_DIAG);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	std::pair<string, string> id_pair(sensor_id, c_sid);
	snapshot->add_sensor(id_pair, diag, w, d);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Sensor created"));
	response.set_body(message);
}

void SensorHandler::delete_sensor(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	string sensor_id = get_string_input(data, UMA_SENSOR_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->delete_sensor(sensor_id);

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Sensor deleted"));
	response.set_body(message);
}

SensorHandler::~SensorHandler() {}