#include "SensorHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "UMAException.h"

SensorHandler::SensorHandler(const string &handler_name) :UMARestHandler(handler_name) {
}

void SensorHandler::handle_create(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/sensor") {
		create_sensor(request);
		return;
	}

	throw UMAException("Cannot handle POST " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::handle_update(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	throw UMAException("Cannot handle PUT " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::handle_read(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/sensor") {
		get_sensor(request);
		return;
	}
	else if (request_url == "/UMA/object/sensor_pair") {
		get_sensor_pair(request);
		return;
	}

	throw UMAException("Cannot handle GET " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::handle_delete(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/sensor") {
		delete_sensor(request);
		return;
	}

	throw UMAException("Cannot handle DELETE " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::get_sensor(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string sensor_id = request.get_string_query("sensor_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	Sensor *sensor = snapshot->getSensor(sensor_id);

	const int idx = sensor->getIdx();
	vector<int> amper_list_idx = sensor->getAmperList();
	vector<bool> amper_list = snapshot->getAmperList(sensor_id);
	vector<string> amper_list_ids = snapshot->getAmperListID(sensor_id);

	request.set_message("Sensor info get");
	request.set_data("amper_list_idx", amper_list_idx);
	request.set_data("amper_list", amper_list);
	request.set_data("amper_list_ids", amper_list_ids);
	request.set_data("idx", idx);
}

void SensorHandler::get_sensor_pair(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string sensor_id1 = request.get_string_query("sensor1");
	const string sensor_id2 = request.get_string_query("sensor2");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	Sensor *sensor1 = snapshot->getSensor(sensor_id1);
	Sensor *sensor2 = snapshot->getSensor(sensor_id2);
	SensorPair *sensor_pair = snapshot->getSensorPair(sensor1, sensor2);
	const double threshold = sensor_pair->getThreshold();

	request.set_message("Sensor Pair info get");
	request.set_data("threshold", threshold);
}

void SensorHandler::create_sensor(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string sensor_id = request.get_string_data("sensor_id");
	const string c_sid = request.get_string_data("c_sid");

	vector<vector<double> > w = request.get_double2d_data("w");
	vector<vector<bool> > d = request.get_bool2d_data("d");
	vector<double> diag = request.get_double1d_data("diag");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	std::pair<string, string> id_pair(sensor_id, c_sid);
	snapshot->add_sensor(id_pair, diag, w, d);

	request.set_message("Sensor created");
}

void SensorHandler::delete_sensor(UMARestRequest &request) {
	const string agent_id = request.get_string_data("agent_id");
	const string snapshot_id = request.get_string_data("snapshot_id");
	const string sensor_id = request.get_string_data("sensor_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->delete_sensor(sensor_id);

	request.set_message("Sensor deleted");
}

SensorHandler::~SensorHandler() {}