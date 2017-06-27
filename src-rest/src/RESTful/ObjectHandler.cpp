#include "ObjectHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"

ObjectHandler::ObjectHandler(logManager *log_access) :AdminHandler(log_access) {
	UMA_BASE_SENSOR_SIZE = L"base_sensor_size";
	UMA_THRESHOLD = L"threshold";
	UMA_Q = L"q";
}

void ObjectHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
	json::value &data = request.extract_json().get();
	if (paths[0] == UMA_AGENT) {
		create_agent(world, data, request);
		return;
	}
	else if (paths[0] == UMA_SNAPSHOT) {
		create_snapshot(world, data, request);
		return;
	}
	else if (paths[0] == UMA_SENSOR) {
		create_sensor(world, data, request);
		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
	json::value message;
	message[MESSAGE] = json::value::string(L"cannot handle " + paths[0] + L" object");
	request.reply(status_codes::BadRequest, message);
}

void ObjectHandler::handle_update(World *world, vector<string_t> &paths, http_request &request) {
	json::value &data = request.extract_json().get();
	if (paths[0] == UMA_AGENT) {
		update_agent(world, data, request);
		return;
	}
	else if (paths[0] == UMA_SNAPSHOT) {
		update_snapshot(world, data, request);
		return;
	}
	else if (paths[0] == UMA_SENSOR) {

		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
	json::value message;
	message[MESSAGE] = json::value::string(L"cannot handle " + paths[0] + L" object");
	request.reply(status_codes::BadRequest, message);
}

void ObjectHandler::handle_read(World *world, vector<string_t> &paths, http_request &request) {

}

void ObjectHandler::handle_delete(World *world, vector<string_t> &paths, http_request &request) {

}

void ObjectHandler::create_agent(World *world, json::value &data, http_request &request) {
	string name, uuid;
	try {
		name = get_string_input(data, NAME, request);
		uuid = get_string_input(data, UUID, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	bool status = world->add_agent(name, uuid);
	if (status) {
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		json::value message;
		message[MESSAGE] = json::value::string(L"Agent created");
		request.reply(status_codes::Created, message);
	}
	else {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"Cannot create agent!");
		request.reply(status_codes::BadRequest, message);
	}
}

void ObjectHandler::create_snapshot(World *world, json::value &data, http_request &request) {
	string name, uuid, agent_id;
	try {
		name = get_string_input(data, NAME, request);
		uuid = get_string_input(data, UUID, request);
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	bool status = agent->add_snapshot_stationary(name, uuid);
	if (status) {
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		json::value message;
		message[MESSAGE] = json::value::string(L"Snapshot created");
		request.reply(status_codes::Created, message);
	}
	else {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"Cannot create snapshot");
		request.reply(status_codes::BadRequest, message);
	}
}

void ObjectHandler::create_sensor(World *world, json::value &data, http_request &request) {
	string name, uuid, agent_id, snapshot_id;
	try {
		name = get_string_input(data, NAME, request);
		uuid = get_string_input(data, UUID, request);
		agent_id = get_string_input(data, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(data, UMA_SNAPSHOT_ID, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	Snapshot *snapshot = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	if (!get_snapshot_by_id(agent, snapshot_id, snapshot, request)) return;
	bool status = snapshot->add_sensor(name, uuid);
	if (status) {
		_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
		json::value message;
		message[MESSAGE] = json::value::string(L"Sensor created");
		request.reply(status_codes::Created, message);
	}
	else {
		_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
		json::value message;
		message[MESSAGE] = json::value::string(L"Cannot create sensor");
		request.reply(status_codes::BadRequest, message);
	}
}

void ObjectHandler::update_agent(World *world, json::value &data, http_request &request) {
	std::map<string_t, string_t> &query = uri::split_query(request.request_uri().query());
	string agent_id;
	try {
		agent_id = get_string_input(query, UMA_AGENT_ID, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	if (check_field(data, NAME, request, false)) {
		try {
			string name = get_string_input(data, NAME, request);
			agent->setName(name);
			_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
			json::value message;
			message[MESSAGE] = json::value::string(L"agent name changed to " + data[NAME].as_string());
			request.reply(status_codes::OK, message);
		}
		catch (exception &e) {
			cout << e.what() << endl;
		}
		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
	json::value message;
	message[MESSAGE] = json::value::string(L"not a valid field to update!");
	request.reply(status_codes::BadRequest, message);
}

void ObjectHandler::update_snapshot(World *world, json::value &data, http_request &request) {
	std::map<string_t, string_t> &query = uri::split_query(request.request_uri().query());
	string agent_id, snapshot_id;
	try {
		agent_id = get_string_input(query, UMA_AGENT_ID, request);
		snapshot_id = get_string_input(query, UMA_SNAPSHOT_ID, request);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		return;
	}

	Agent *agent = NULL;
	Snapshot *snapshot = NULL;
	if (!get_agent_by_id(world, agent_id, agent, request)) return;
	if (!get_snapshot_by_id(agent, snapshot_id, snapshot, request)) return;

	if (check_field(data, NAME, request, false)) {
		try{
			string name = get_string_input(data, NAME, request);
			snapshot->setName(name);
			_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
			json::value message;
			message[MESSAGE] = json::value::string(L"snapshot name changed to " + data[NAME].as_string());
			request.reply(status_codes::OK, message);
		}
		catch (exception &e) {
			cout << e.what() << endl;
		}
		return;
	}
	else if (check_field(data, UMA_Q, request, false)) {
		try {
			double q = get_double_input(data, UMA_Q, request);
			snapshot->setQ(q);
			_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
			json::value message;
			message[MESSAGE] = json::value::string(L"snapshot q changed to " + to_wstring(q));
			request.reply(status_codes::OK, message);
		}
		catch (exception &e) {
			cout << e.what() << endl;
		}
		return;
	}
	else if (check_field(data, UMA_THRESHOLD, request, false)) {
		try {
			double threshold = get_double_input(data, UMA_THRESHOLD, request);
			snapshot->setThreshold(threshold);
			_log_access->info() << REQUEST_MODE + request.absolute_uri().to_string() + L" 201";
			json::value message;
			message[MESSAGE] = json::value::string(L"snapshot threshold changed to " + to_wstring(threshold));
			request.reply(status_codes::OK, message);
		}
		catch (exception &e) {
			cout << e.what() << endl;
		}
		return;
	}

	_log_access->error() << REQUEST_MODE + request.absolute_uri().to_string() + L" 400";
	json::value message;
	message[MESSAGE] = json::value::string(L"not a valid field to update!");
	request.reply(status_codes::BadRequest, message);
}

ObjectHandler::~ObjectHandler() {}