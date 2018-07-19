#include "SensorHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "UMAException.h"

SensorHandler::SensorHandler(const string &handler_name) :UMARestHandler(handler_name) {
}

void SensorHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/sensor") {
		createSensor(request);
		return;
	}

	throw UMAException("Cannot handle POST " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	throw UMAException("Cannot handle PUT " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/sensor") {
		getSensor(request);
		return;
	}
	else if (requestUrl == "/UMA/object/sensorPair") {
		getSensorPair(request);
		return;
	}

	throw UMAException("Cannot handle GET " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.get_request_url();
	if (requestUrl == "/UMA/object/sensor") {
		deleteSensor(request);
		return;
	}

	throw UMAException("Cannot handle DELETE " + requestUrl, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void SensorHandler::getSensor(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");
	const string sensorId = request.get_string_query("sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	Sensor *sensor = snapshot->getSensor(sensorId);

	const int idx = sensor->getIdx();
	vector<int> amperListIdx = sensor->getAmperList();
	vector<bool> amperList = snapshot->getAmperList(sensorId);
	vector<string> amperListIds = snapshot->getAmperListID(sensorId);

	request.set_message("Sensor info get");
	request.set_data("amper_list_idx", amperListIdx);
	request.set_data("amper_list", amperList);
	request.set_data("amper_list_ids", amperListIds);
	request.set_data("idx", idx);
}

void SensorHandler::getSensorPair(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");
	const string sensorId1 = request.get_string_query("sensor1");
	const string sensorId2 = request.get_string_query("sensor2");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	Sensor *sensor1 = snapshot->getSensor(sensorId1);
	Sensor *sensor2 = snapshot->getSensor(sensorId2);
	SensorPair *sensorPair = snapshot->getSensorPair(sensor1, sensor2);
	const double threshold = sensorPair->getThreshold();

	request.set_message("Sensor Pair info get");
	request.set_data("threshold", threshold);
}

void SensorHandler::createSensor(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const string sensorId = request.get_string_data("sensor_id");
	const string cSid = request.get_string_data("c_sid");

	vector<vector<double> > w = request.get_double2d_data("w");
	vector<vector<bool> > d = request.get_bool2d_data("d");
	vector<double> diag = request.get_double1d_data("diag");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	std::pair<string, string> idPair(sensorId, cSid);
	snapshot->createSensor(idPair, diag, w, d);

	request.set_message("Sensor created");
}

void SensorHandler::deleteSensor(UMARestRequest &request) {
	const string experimentId = request.get_string_data("experiment_id");
	const string agentId = request.get_string_data("agent_id");
	const string snapshotId = request.get_string_data("snapshot_id");
	const string sensorId = request.get_string_data("sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->deleteSensor(sensorId);

	request.set_message("Sensor deleted");
}

SensorHandler::~SensorHandler() {}