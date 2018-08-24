#include "AttrSensorHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"

static Logger serverLogger("Server", "log/UMA_server.log");

AttrSensorHandler::AttrSensorHandler(const string &handlerName) : UMARestHandler(handlerName) {
}

void AttrSensorHandler::handleCreate(UMARestRequest &request) {
	//no post call available for attr_sensor or attr_sensor pair, as they are created by sensor
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void AttrSensorHandler::handleUpdate(UMARestRequest &request) {
	//only for test purpose
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/attrSensor") {
		updateAttrSensor(request);
		return;
	}
	else if (requestUrl == "/UMA/object/attrSensorPair") {
		updateAttrSensorPair(request);
		return;
	}
	throw UMABadOperationException("Cannot handle PUT " + requestUrl, false, &serverLogger);
}

void AttrSensorHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	if (requestUrl == "/UMA/object/attrSensor") {
		getAttrSensor(request);
		return;
	}
	else if (requestUrl == "/UMA/object/attrSensorPair") {
		getAttrSensorPair(request);
		return;
	}

	throw UMABadOperationException("Cannot handle GET " + requestUrl, false, &serverLogger);
}

void AttrSensorHandler::handleDelete(UMARestRequest &request) {
	//no delete call for attrSensor attrSensorPair, they are handled in sensor
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle DELETE " + requestUrl, false, &serverLogger);
}

void AttrSensorHandler::getAttrSensor(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");
	const string attrSensorId = request.getStringQuery("attr_sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensor *attrSensor = snapshot->getAttrSensor(attrSensorId);
	const double diag = attrSensor->getDiag();
	const double oldDiag = attrSensor->getOldDiag();
	bool current = attrSensor->getCurrent();
	bool isOriginPure = attrSensor->getIsOriginPure();

	request.setMessage("AttrSensor info get");
	request.setData("diag", diag);
	request.setData("old_diag", oldDiag);
	request.setData("status", current);
	request.setData("is_origin_pure", isOriginPure);
}

void AttrSensorHandler::updateAttrSensor(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");
	const string attrSensorId = request.getStringQuery("attr_sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensor *attrSensor = snapshot->getAttrSensor(attrSensorId);

	if (request.checkDataField("diag")) {
		const double diag = request.getDoubleData("diag");
		attrSensor->setDiag(diag);

		request.setMessage("Diag updated");
		return;
	}
	else if (request.checkDataField("old_diag")) {
		const double oldDiag = request.getDoubleData("old_diag");
		attrSensor->setOldDiag(oldDiag);

		request.setMessage("Old diag updated");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

void AttrSensorHandler::getAttrSensorPair(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");
	const string attrSensorId1 = request.getStringQuery("attr_sensor1");
	const string attrSensorId2 = request.getStringQuery("attr_sensor2");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensorPair *attrSensorPair = snapshot->getAttrSensorPair(attrSensorId1, attrSensorId2);

	const double w = attrSensorPair->getW();
	const bool d = attrSensorPair->getD();
	request.setMessage("AttrSensor pair info get");
	request.setData("w", w);
	request.setData("d", d);
}

void AttrSensorHandler::updateAttrSensorPair(UMARestRequest &request) {
	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");
	const string attrSensorId1 = request.getStringQuery("attr_sensor1");
	const string attrSensorId2 = request.getStringQuery("attr_sensor2");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensorPair *attrSensorPair = snapshot->getAttrSensorPair(attrSensorId1, attrSensorId2);

	if (request.checkDataField("w")) {
		const double w = request.getDoubleData("w");
		attrSensorPair->setW(w);

		request.setMessage("w updated");
		return;
	}
	else if (request.checkDataField("d")) {
		const bool d = request.getBoolData("d");
		attrSensorPair->setD(d);

		request.setMessage("d updated");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

AttrSensorHandler::~AttrSensorHandler() {}