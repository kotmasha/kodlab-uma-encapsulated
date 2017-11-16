#include "MeasurableHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "Measurable.h"
#include "MeasurablePair.h"

MeasurableHandler::MeasurableHandler(string handler_factory) : AdminHandler(handler_factory) {
	UMA_MEASURABLE1 = U("measurable1");
	UMA_MEASURABLE2 = U("measurable2");
	UMA_W = U("w");
	UMA_D = U("d");
	UMA_DIAG = U("diag");
	UMA_OLD_DIAG = U("old_diag");
}

void MeasurableHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	//no post call available for measurable or measurable pair, as they are created by sensor
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void MeasurableHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	//only for test purpose
	if (path == U("/UMA/object/measurable")) {
		update_measurable(world, request, response);
		return;
	}
	else if (path == U("/UMA/object/measurable_pair")) {
		update_measurable_pair(world, request, response);
		return;
	}
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void MeasurableHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/measurable")) {
		get_measurable(world, request, response);
		return;
	}
	else if (path == U("/UMA/object/measurable_pair")) {
		get_measurable_pair(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void MeasurableHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {
	//no delete call for measurable measurable_pair, they are handled in snesor
	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::CLIENT_ERROR, status_codes::BadRequest);
}

void MeasurableHandler::get_measurable(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string measurable_id = get_string_input(query, UMA_MEASURABLE_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	Measurable *measurable = snapshot->getMeasurable(measurable_id);
	double diag = measurable->getDiag();
	double old_diag = measurable->getOldDiag();
	bool current = measurable->getCurrent();
	bool isOriginPure = measurable->getIsOriginPure();

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Measurable info get"));
	message[DATA] = json::value();
	message[DATA][U("diag")] = json::value(diag);
	message[DATA][U("old_diag")] = json::value(old_diag);
	message[DATA][U("status")] = json::value(current);
	message[DATA][U("isOriginPure")] = json::value(isOriginPure);
	response.set_body(message);
}

void MeasurableHandler::update_measurable(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string measurable_id = get_string_input(query, UMA_MEASURABLE_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	Measurable *measurable = snapshot->getMeasurable(measurable_id);

	if (check_field(data, UMA_DIAG, false)) {
		double diag = get_double_input(data, UMA_DIAG);
		measurable->setDiag(diag);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Diag updated"));
		response.set_body(message);
		return;
	}
	else if (check_field(data, UMA_OLD_DIAG, false)) {
		double old_diag = get_double_input(data, UMA_OLD_DIAG);
		measurable->setOldDiag(old_diag);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Old diag updated"));
		response.set_body(message);
		return;
	}

	throw ClientException("The coming put request has nothing to update", ClientException::CLIENT_ERROR, status_codes::NotAcceptable);
}

void MeasurableHandler::get_measurable_pair(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string measurable_id1 = get_string_input(query, UMA_MEASURABLE1);
	string measurable_id2 = get_string_input(query, UMA_MEASURABLE2);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	MeasurablePair *measurable_pair = snapshot->getMeasurablePair(measurable_id1, measurable_id2);

	double w = measurable_pair->getW();
	bool d = measurable_pair->getD();
	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Measurable pair info get"));
	message[DATA] = json::value();
	message[DATA][U("w")] = json::value(w);
	message[DATA][U("d")] = json::value(d);
	response.set_body(message);
}

void MeasurableHandler::update_measurable_pair(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string measurable_id1 = get_string_input(query, UMA_MEASURABLE1);
	string measurable_id2 = get_string_input(query, UMA_MEASURABLE2);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	MeasurablePair *measurable_pair = snapshot->getMeasurablePair(measurable_id1, measurable_id2);

	if (check_field(data, UMA_W, false)) {
		double w = get_double_input(data, UMA_W);
		measurable_pair->setW(w);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("W updated"));
		response.set_body(message);
		return;
	}
	else if (check_field(data, UMA_D, false)) {
		bool d = get_bool_input(data, UMA_D);
		measurable_pair->setD(d);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("D updated"));
		response.set_body(message);
		return;
	}

	throw ClientException("The coming put request has nothing to update", ClientException::CLIENT_ERROR, status_codes::NotAcceptable);
}

MeasurableHandler::~MeasurableHandler() {}