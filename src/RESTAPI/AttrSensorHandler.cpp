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

AttrSensorHandler::AttrSensorHandler(const string &handler_name) : UMARestHandler(handler_name) {
}

void AttrSensorHandler::handleCreate(UMARestRequest &request) {
	//no post call available for attr_sensor or attr_sensor pair, as they are created by sensor
	const string requestUrl = request.get_request_url();
	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void AttrSensorHandler::handleUpdate(UMARestRequest &request) {
	//only for test purpose
	const string requestUrl = request.get_request_url();
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
	const string requestUrl = request.get_request_url();
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
	const string requestUrl = request.get_request_url();
	throw UMABadOperationException("Cannot handle DELETE " + requestUrl, false, &serverLogger);
}

void AttrSensorHandler::getAttrSensor(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");
	const string attrSensorId = request.get_string_query("attr_sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensor *attrSensor = snapshot->getAttrSensor(attrSensorId);
	const double diag = attrSensor->getDiag();
	const double old_diag = attrSensor->getOldDiag();
	bool current = attrSensor->getCurrent();
	bool isOriginPure = attrSensor->getIsOriginPure();

	request.set_message("AttrSensor info get");
	request.set_data("diag", diag);
	request.set_data("old_diag", old_diag);
	request.set_data("status", current);
	request.set_data("is_origin_pure", isOriginPure);
}

void AttrSensorHandler::updateAttrSensor(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");
	const string attrSensorId = request.get_string_query("attr_sensor_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensor *attrSensor = snapshot->getAttrSensor(attrSensorId);

	if (request.check_data_field("diag")) {
		const double diag = request.get_double_data("diag");
		attrSensor->setDiag(diag);

		request.set_message("Diag updated");
		return;
	}
	else if (request.check_data_field("old_diag")) {
		const double old_diag = request.get_double_data("old_diag");
		attrSensor->setOldDiag(old_diag);

		request.set_message("Old diag updated");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

void AttrSensorHandler::getAttrSensorPair(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");
	const string attrSensorId1 = request.get_string_query("attr_sensor1");
	const string attrSensorId2 = request.get_string_query("attr_sensor2");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensorPair *attrSensorPair = snapshot->getAttrSensorPair(attrSensorId1, attrSensorId2);

	const double w = attrSensorPair->getW();
	const bool d = attrSensorPair->getD();
	request.set_message("AttrSensor pair info get");
	request.set_data("w", w);
	request.set_data("d", d);
}

void AttrSensorHandler::updateAttrSensorPair(UMARestRequest &request) {
	const string experimentId = request.get_string_query("experiment_id");
	const string agentId = request.get_string_query("agent_id");
	const string snapshotId = request.get_string_query("snapshot_id");
	const string attrSensorId1 = request.get_string_query("attr_sensor1");
	const string attrSensorId2 = request.get_string_query("attr_sensor2");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	AttrSensorPair *attrSensorPair = snapshot->getAttrSensorPair(attrSensorId1, attrSensorId2);

	if (request.check_data_field("w")) {
		const double w = request.get_double_data("w");
		attrSensorPair->setW(w);

		request.set_message("w updated");
		return;
	}
	else if (request.check_data_field("d")) {
		const bool d = request.get_bool_data("d");
		attrSensorPair->setD(d);

		request.set_message("d updated");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

AttrSensorHandler::~AttrSensorHandler() {}