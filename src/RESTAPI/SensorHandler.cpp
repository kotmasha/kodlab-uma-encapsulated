#include "SensorHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "UMAException.h"

static Logger serverLogger("Server", "log/UMA_server.log");
SensorHandler::SensorHandler(const string &handlerName) :UMARestHandler(handlerName) {
}

void SensorHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/sensor") {
		createSensor(request);
		return;
	}

	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void SensorHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle PUT " + requestUrl, false, &serverLogger);
}

void SensorHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/sensor") {
		getSensor(request);
		return;
	}
	else if (requestUrl == "/UMA/object/sensorPair") {
		getSensorPair(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void SensorHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/sensor") {
		deleteSensor(request);
		return;
	}

	throw UMABadOperationException("Cannot handle DELETE " + requestUrl, false, &serverLogger);
}

void SensorHandler::getSensor(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");
	const string sensorId = request.getStringQuery("sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	Sensor *sensor = snapshot->getSensor(sensorId);

	const int idx = sensor->getIdx();
	vector<int> amperListIdx = sensor->getAmperList();
	vector<bool> amperList = snapshot->getAmperList(sensorId);
	vector<string> amperListIds = snapshot->getAmperListID(sensorId);

	request.setMessage("Sensor info get");
	request.setData("amper_list_idx", amperListIdx);
	request.setData("amper_list", amperList);
	request.setData("amper_list_ids", amperListIds);
	request.setData("idx", idx);
}

void SensorHandler::getSensorPair(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");
	const string sensorId1 = request.getStringQuery("sensor1");
	const string sensorId2 = request.getStringQuery("sensor2");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	Sensor *sensor1 = snapshot->getSensor(sensorId1);
	Sensor *sensor2 = snapshot->getSensor(sensorId2);
	SensorPair *sensorPair = snapshot->getSensorPair(sensor1, sensor2);
	const double threshold = sensorPair->getThreshold();

	request.setMessage("Sensor Pair info get");
	request.setData("threshold", threshold);
}

void SensorHandler::createSensor(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const string sensorId = request.getStringData("sensor_id");
	const string cSid = request.getStringData("c_sid");

	vector<vector<double> > w = request.getDouble2dData("w");
	vector<vector<bool> > d = request.getBool2dData("d");
	vector<double> diag = request.getDouble1dData("diag");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	std::pair<string, string> idPair(sensorId, cSid);
	snapshot->createSensor(idPair, diag, w, d);

	request.setMessage("Sensor created");
}

void SensorHandler::deleteSensor(UMARestRequest &request) {
	const string experimentId = request.getStringData("experiment_id");
	const string agentId = request.getStringData("agent_id");
	const string snapshotId = request.getStringData("snapshot_id");
	const string sensorId = request.getStringData("sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	snapshot->deleteSensor(sensorId);

	request.setMessage("Sensor deleted");
}

SensorHandler::~SensorHandler() {}