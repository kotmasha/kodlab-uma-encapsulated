#include "DataHandler.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "logManager.h"

DataHandler::DataHandler(logManager *log_access):AdminHandler(log_access) {
	UMA_BASE_SENSOR_SIZE = L"base_sensor_size";
	UMA_THRESHOLD = L"threshold";
	UMA_Q = L"q";
}

void DataHandler::handle_create(World *world, vector<string_t> &paths, http_request &request) {
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

	_log_access->error() << request.absolute_uri().to_string() + L" 400";
	request.reply(status_codes::BadRequest, json::value::string(L"cannot handle " + paths[0] + L" object"));
}

void DataHandler::handle_update(World *world, vector<string_t> &paths, http_request &request) {

}

void DataHandler::handle_read(World *world, vector<string_t> &paths, http_request &request) {

}

void DataHandler::handle_delete(World *world, vector<string_t> &paths, http_request &request) {

}

void DataHandler::create_agent(World *world, json::value &data, http_request &request) {
	if (!check_field(data, NAME, request)) return;
	if (!check_field(data, UUID, request)) return;
	string_t name, uuid;
	try {
		name = data[NAME].as_string();
		uuid = data[UUID].as_string();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		return;
	}
	//convent from wstring to string
	std::string s_name(name.begin(), name.end());
	std::string s_uuid(uuid.begin(), uuid.end());
	bool status = world->add_agent(s_name, s_uuid);
	if (status) {
		_log_access->info()<< request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, json::value::string(L"Agent created"));
	}
	else {
		_log_access->error()<< request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"Cannot create agent!"));
	}
}

void DataHandler::create_snapshot(World *world, json::value &data, http_request &request) {
	if (!check_field(data, NAME, request)) return;
	if (!check_field(data, UUID, request)) return;
	if (!check_field(data, UMA_AGENT_ID, request)) return;

	string_t name, uuid, agent_id;
	try {
		name = data[NAME].as_string();
		uuid = data[UUID].as_string();
		agent_id = data[UMA_AGENT_ID].as_string();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		return;
	}
	//convent from wstring to string
	std::string s_name(name.begin(), name.end());
	std::string s_uuid(uuid.begin(), uuid.end());
	std::string s_agent_id(agent_id.begin(), agent_id.end());
	
	Agent *agent = world->getAgent(s_agent_id);
	if (agent == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		request.reply(status_codes::NotFound, json::value::string(L"Cannot find the agent id!"));
		return;
	}
	bool status = agent->add_snapshot_stationary(s_name, s_uuid);
	if (status) {
		_log_access->info() << request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, json::value::string(L"Snapshot created"));
	}
	else {
		_log_access->error() << request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"Cannot create snapshot!"));
	}
}

void DataHandler::create_sensor(World *world, json::value &data, http_request &request) {
	if (!check_field(data, NAME, request)) return;
	if (!check_field(data, UUID, request)) return;
	if (!check_field(data, UMA_AGENT_ID, request)) return;
	if (!check_field(data, UMA_SNAPSHOT_ID, request)) return;

	string_t name, uuid, agent_id, snapshot_id;
	try {
		name = data[NAME].as_string();
		uuid = data[UUID].as_string();
		agent_id = data[UMA_AGENT_ID].as_string();
		snapshot_id = data[UMA_SNAPSHOT_ID].as_string();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		return;
	}
	//convent from wstring to string
	std::string s_name(name.begin(), name.end());
	std::string s_uuid(uuid.begin(), uuid.end());
	std::string s_agent_id(agent_id.begin(), agent_id.end());
	std::string s_snapshot_id(snapshot_id.begin(), snapshot_id.end());

	Agent *agent = world->getAgent(s_agent_id);
	if (agent == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		request.reply(status_codes::NotFound, json::value::string(L"Cannot find the agent id!"));
		return;
	}
	Snapshot *snapshot = agent->getSnapshot(s_snapshot_id);
	if (snapshot == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		request.reply(status_codes::NotFound, json::value::string(L"Cannot find the snapshot id!"));
		return;
	}
	bool status = snapshot->add_sensor(s_name, s_uuid);
	if (status) {
		_log_access->info() << request.absolute_uri().to_string() + L" 201";
		request.reply(status_codes::Created, json::value::string(L"Sensor created"));
	}
	else {
		_log_access->error() << request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"Cannot create sensor!"));
	}
}

DataHandler::~DataHandler() {}