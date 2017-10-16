#include "SnapshotHandler.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

SnapshotHandler::SnapshotHandler(string handler_factory, logManager *log_access): AdminHandler(handler_factory, log_access){
	UMA_INITIAL_SIZE = U("initial_size");
	UMA_Q = U("q");
	UMA_THRESHOLD = U("threshold");
	UMA_AUTO_TARGET = U("auto_target");
	UMA_PROPAGATE_MASK = U("propagate_mask");
	UMA_FROM_SENSOR = U("from_sensor");
	UMA_TO_SENSOR = U("to_sensor");
}

void SnapshotHandler::handle_create(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/snapshot")) {
		create_snapshot(world, request, response);
		return;
	}
	else if (path == U("/UMA/object/snapshot/implication")) {
		create_implication(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SnapshotHandler::handle_update(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/snapshot")) {
		update_snapshot(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SnapshotHandler::handle_read(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/snapshot")) {
		get_snapshot(world, request, response);
		return;
	}
	else if (path == U("/UMA/object/snapshot/implication")) {
		get_implication(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SnapshotHandler::handle_delete(World *world, string_t &path, http_request &request, http_response &response) {
	if (path == U("/UMA/object/snapshot")) {
		delete_snapshot(world, request, response);
		return;
	}
	else if (path == U("/UMA/object/snapshot/implication")) {
		delete_implication(world, request, response);
		return;
	}

	throw ClientException("Cannot handle " + string_t_to_string(path), ClientException::ERROR, status_codes::BadRequest);
}

void SnapshotHandler::create_snapshot(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string uuid = get_string_input(data, UMA_SNAPSHOT_ID);
	string agent_id = get_string_input(data, UMA_AGENT_ID);

	Agent *agent = world->getAgent(agent_id);
	agent->add_snapshot_stationary(uuid);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Snapshot created"));
	response.set_body(message);
}

void SnapshotHandler::create_implication(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string from_sensor = get_string_input(data, UMA_FROM_SENSOR);
	string to_sensor = get_string_input(data, UMA_TO_SENSOR);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->create_implication(from_sensor, to_sensor);

	response.set_status_code(status_codes::Created);
	json::value message;
	message[MESSAGE] = json::value::string(U("Implication created"));
	response.set_body(message);
}

void SnapshotHandler::get_snapshot(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);

	vector<std::pair<int, pair<string, string> > > sensor_info = snapshot->getSensorInfo();
	double total = snapshot->getTotal();
	double q = snapshot->getQ();
	double threshold = snapshot->getThreshold();
	bool auto_target = snapshot->getAutoTarget();
	bool propagate_mask = snapshot->getPropagateMask();
	int initial_size = snapshot->getInitialSize();

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Get snapshot info"));
	message[DATA] = json::value();
	json::value converted_sensor_info = convert_sensor_info(sensor_info);
	message[DATA][U("sensors")] = converted_sensor_info;

	message[DATA][U("total")] = total;
	message[DATA][U("q")] = q;
	message[DATA][U("threshold")] = threshold;
	message[DATA][U("auto_target")] = auto_target;
	message[DATA][U("propagate_mask")] = propagate_mask;
	message[DATA][U("initial_size")] = initial_size;
	response.set_body(message);
}

void SnapshotHandler::get_implication(World *world, http_request &request, http_response &response) {
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());
	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);
	string from_sensor = get_string_input(query, UMA_FROM_SENSOR);
	string to_sensor = get_string_input(query, UMA_TO_SENSOR);
	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	bool value = snapshot->get_implication(from_sensor, to_sensor);

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Implication created"));
	message[DATA] = json::value();
	message[DATA][U("implication")] = value;
	response.set_body(message);
}

void SnapshotHandler::delete_snapshot(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	agent->delete_snapshot(snapshot_id);
	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Snapshot deleted"));
	response.set_body(message);
}

void SnapshotHandler::delete_implication(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	string agent_id = get_string_input(data, UMA_AGENT_ID);
	string snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID);
	string from_sensor = get_string_input(data, UMA_FROM_SENSOR);
	string to_sensor = get_string_input(data, UMA_TO_SENSOR);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);
	snapshot->delete_implication(from_sensor, to_sensor);

	response.set_status_code(status_codes::OK);
	json::value message;
	message[MESSAGE] = json::value::string(U("Implication deleted"));
	response.set_body(message);
}

void SnapshotHandler::update_snapshot(World *world, http_request &request, http_response &response) {
	json::value data = request.extract_json().get();
	std::map<string_t, string_t> query = uri::split_query(request.request_uri().query());

	string agent_id = get_string_input(query, UMA_AGENT_ID);
	string snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID);

	Agent *agent = world->getAgent(agent_id);
	Snapshot *snapshot = agent->getSnapshot(snapshot_id);

	if (check_field(data, UMA_Q, false)) {
		double q = get_double_input(data, UMA_Q);
		snapshot->setQ(q);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Q updated"));
		response.set_body(message);
		return;
	}
	else if (check_field(data, UMA_THRESHOLD, false)) {
		double threshold = get_double_input(data, UMA_THRESHOLD);
		snapshot->setThreshold(threshold);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Threshold updated"));
		response.set_body(message);
		return;
	}
	else if (check_field(data, UMA_AUTO_TARGET, false)) {
		bool auto_target = get_bool_input(data, UMA_AUTO_TARGET);
		snapshot->setAutoTarget(auto_target);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Auto target updated"));
		response.set_body(message);
		return;
	}
	else if (check_field(data, UMA_PROPAGATE_MASK, false)) {
		bool propagate_mask = get_bool_input(data, UMA_PROPAGATE_MASK);
		snapshot->setPropagateMask(propagate_mask);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("Auto target updated"));
		response.set_body(message);
		return;
	}
	else if (check_field(data, UMA_INITIAL_SIZE, false)) {
		int initial_size = get_int_input(data, UMA_INITIAL_SIZE);
		snapshot->setInitialSize(initial_size);

		response.set_status_code(status_codes::OK);
		json::value message;
		message[MESSAGE] = json::value::string(U("initial size updated"));
		response.set_body(message);
		return;
	}

	throw ClientException("The coming put request has nothing to update", ClientException::ERROR, status_codes::NotAcceptable);
}

json::value SnapshotHandler::convert_sensor_info(const vector<std::pair<int, pair<string, string> > > &sensor_info) {
	json::value sensors;
	for (int i = 0; i < sensor_info.size(); ++i) {
		json::value sensor, measurables;
		auto it = sensor_info[i];
		measurables[U("m")] = json::value(string_to_string_t(it.second.first));
		measurables[U("cm")] = json::value(string_to_string_t(it.second.second));
		sensor[U("sensor")] = measurables;
		sensors[it.first] = sensor;
	}
	return sensors;
}

SnapshotHandler::~SnapshotHandler() {}