#include "AttrSensorHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"

AttrSensorHandler::AttrSensorHandler(const string &handler_name) : UMARestHandler(handler_name) {
}

void AttrSensorHandler::handle_create(UMARestRequest &request) {
	//no post call available for measurable or measurable pair, as they are created by sensor
	const string request_url = request.get_request_url();
	throw UMAException("Cannot handle POST " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AttrSensorHandler::handle_update(UMARestRequest &request) {
	//only for test purpose
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/measurable") {
		update_measurable(request);
		return;
	}
	else if (request_url == "/UMA/object/measurable_pair") {
		update_measurable_pair(request);
		return;
	}
	throw UMAException("Cannot handle PUT " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AttrSensorHandler::handle_read(UMARestRequest &request) {
	const string request_url = request.get_request_url();
	if (request_url == "/UMA/object/measurable") {
		get_measurable(request);
		return;
	}
	else if (request_url == "/UMA/object/measurable_pair") {
		get_measurable_pair(request);
		return;
	}

	throw UMAException("Cannot handle GET " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AttrSensorHandler::handle_delete(UMARestRequest &request) {
	//no delete call for measurable measurable_pair, they are handled in sensor
	const string request_url = request.get_request_url();
	throw UMAException("Cannot handle DELETE " + request_url, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AttrSensorHandler::get_measurable(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string measurable_id = request.get_string_query("measurable_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	AttrSensor *measurable = snapshot->getAttrSensor(measurable_id);
	const double diag = measurable->getDiag();
	const double old_diag = measurable->getOldDiag();
	bool current = measurable->getCurrent();
	bool isOriginPure = measurable->getIsOriginPure();

	request.set_message("AttrSensor info get");
	request.set_data("diag", diag);
	request.set_data("old_diag", old_diag);
	request.set_data("status", current);
	request.set_data("isOriginPure", isOriginPure);
}

void AttrSensorHandler::update_measurable(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string measurable_id = request.get_string_query("measurable_id");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	AttrSensor *measurable = snapshot->getAttrSensor(measurable_id);

	if (request.check_data_field("diag")) {
		const double diag = request.get_double_data("diag");
		measurable->setDiag(diag);

		request.set_message("Diag updated");
		return;
	}
	else if (request.check_data_field("old_diag")) {
		const double old_diag = request.get_double_data("old_diag");
		measurable->setOldDiag(old_diag);

		request.set_message("Old diag updated");
		return;
	}

	throw UMAException("The coming put request has nothing to update", UMAException::ERROR_LEVEL::WARN, UMAException::ERROR_TYPE::BAD_OPERATION);
}

void AttrSensorHandler::get_measurable_pair(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string measurable_id1 = request.get_string_query("measurable1");
	const string measurable_id2 = request.get_string_query("measurable2");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	AttrSensorPair *measurable_pair = snapshot->getAttrSensorPair(measurable_id1, measurable_id2);

	const double w = measurable_pair->getW();
	const bool d = measurable_pair->getD();
	request.set_message("AttrSensor pair info get");
	request.set_data("w", w);
	request.set_data("d", d);
}

void AttrSensorHandler::update_measurable_pair(UMARestRequest &request) {
	const string agent_id = request.get_string_query("agent_id");
	const string snapshot_id = request.get_string_query("snapshot_id");
	const string measurable_id1 = request.get_string_query("measurable1");
	const string measurable_id2 = request.get_string_query("measurable2");

	Agent *agent = World::getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	AttrSensorPair *measurable_pair = snapshot->getAttrSensorPair(measurable_id1, measurable_id2);

	if (request.check_data_field("w")) {
		const double w = request.get_double_data("w");
		measurable_pair->setW(w);

		request.set_message("w updated");
		return;
	}
	else if (request.check_data_field("d")) {
		const bool d = request.get_bool_data("d");
		measurable_pair->setD(d);

		request.set_message("d updated");
		return;
	}

	throw UMAException("The coming put request has nothing to update", UMAException::ERROR_LEVEL::WARN, UMAException::ERROR_TYPE::BAD_OPERATION);
}

AttrSensorHandler::~AttrSensorHandler() {}