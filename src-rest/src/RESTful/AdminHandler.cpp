#include "AdminHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"

bool to_bool(string_t &s) {
	return s == L"true" || s == L"True";
}

AdminHandler::AdminHandler() {
	UUID = L"uuid";
	UMA_AGENT = L"agent";
	UMA_SNAPSHOT = L"snapshot";
	UMA_SENSOR = L"sensor";
	UMA_SENSOR_PAIR = L"sensor_pair";
	UMA_MEASURABLE = L"measurable";
	UMA_MEASURABLE_PAIR = L"measurable_pair";
	UMA_AGENT_ID = L"agent_id";
	UMA_SNAPSHOT_ID = L"snapshot_id";
	UMA_SENSOR_ID = L"sensor_id";

	GET = L" GET ";
	PUT = L" PUT ";
	POST = L" POST ";
	DELETE = L" DELETE ";

	MESSAGE = L"message";
}

AdminHandler::AdminHandler(logManager *log_access):AdminHandler() {
	_log_access = log_access;
}

bool AdminHandler::check_field(json::value &data, string_t &s, http_request &request, bool hard_check) {
	if (!data.has_field(s)) {
		if (hard_check) {
			_log_access->error() << request.absolute_uri().to_string() + L" 400";
			json::value message;
			message[MESSAGE] = json::value::string(L"Field \'" + s + L"\' must be specified");
			request.reply(status_codes::BadRequest, message);
		}
		return false;
	}
	return true;
}

bool AdminHandler::check_field(map<string_t, string_t> &query, string_t &s, http_request &request, bool hard_check) {
	if (query.find(s) == query.end()) {
		if (hard_check) {
			_log_access->error() << request.absolute_uri().to_string() + L" 400";
			json::value message;
			message[MESSAGE] = json::value::string(L"Field \'" + s + L"\' must be specified");
			request.reply(status_codes::BadRequest, message);
		}
		return false;
	}
	return true;
}

void AdminHandler::vector_bool_to_array(std::vector<bool> &list, std::vector<json::value> &json_list) {
	for (int i = 0; i < list.size(); ++i) {
		json_list.push_back(json::value::boolean(list[i]));
	}
}

void AdminHandler::vector_string_to_array(std::vector<string> &list, std::vector<json::value> &json_list) {
	for (int i = 0; i < list.size(); ++i) {
		string_t tmp(list[i].begin(), list[i].end());
		json_list.push_back(json::value::string(tmp));
	}
}

bool AdminHandler::get_agent_by_id(World *world, string agent_id, Agent *&agent, http_request &request) {
	agent = world->getAgent(agent_id);
	if (agent == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		json::value message;
		message[MESSAGE] = json::value::string(L"Cannot find the agent id!");
		request.reply(status_codes::NotFound, message);
		return false;
	}
	return true;
}

bool AdminHandler::get_snapshot_by_id(Agent *agent, string snapshot_id, Snapshot *&snapshot, http_request &request) {
	snapshot = agent->getSnapshot(snapshot_id);
	if (snapshot == NULL) {
		_log_access->info() << request.absolute_uri().to_string() + L" 404";
		json::value message;
		message[MESSAGE] = json::value::string(L"Cannot find the snapshot id!");
		request.reply(status_codes::NotFound, message);
		return false;
	}
	return true;
}

string AdminHandler::get_string_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	string_t value;
	try {
		value = data[name].as_string();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	std::string s_value(value.begin(), value.end());
	return s_value;
}

string AdminHandler::get_string_input(map<string_t, string_t> &query, string_t &name, http_request &request) {
	if (!check_field(query, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	string_t value;
	try {
		value = query[name];
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	std::string s_value(value.begin(), value.end());
	return s_value;
}

double AdminHandler::get_double_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	double value;
	try {
		value = data[name].as_double();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	return value;
}

double AdminHandler::get_double_input(map<string_t, string_t> &query, string_t &name, http_request &request) {
	if (!check_field(query, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	double value;
	try {
		value = stod(query[name]);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	return value;
}

bool AdminHandler::get_bool_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	bool value;
	try {
		value = data[name].as_bool();
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	return value;
}

bool AdminHandler::get_bool_input(map<string_t, string_t> &query, string_t &name, http_request &request) {
	if (!check_field(query, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	bool value;
	try {
		value = to_bool(query[name]);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	return value;
}

vector<bool> AdminHandler::get_bool1d_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	vector<bool> value;
	try {
		auto &list = data[name].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_bool());
		}
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	return value;
}

vector<vector<bool> > AdminHandler::get_bool2d_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	vector<vector<bool> > value;
	try {
		auto &lists = data[name].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<bool> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(list[j].as_bool());
			}
			value.push_back(tmp_value);
		}
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	return value;
}

vector<string> AdminHandler::get_string1d_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	vector<string> value;
	try {
		auto &list = data[name].as_array();
		for (int i = 0; i < list.size(); ++i) {
			string_t tmp = list[i].as_string();
			std::string s_tmp(tmp.begin(), tmp.end());
			value.push_back(s_tmp);
		}
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	return value;
}

vector<std::pair<string, string> > AdminHandler::get_string_pair1d_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	vector<std::pair<string, string> > value;
	try {
		auto &lists = data[name].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto list = lists[i].as_array();
			if (list.size() != 2) throw CLIENT_EXCEPTION::CLIENT_ERROR;
			string_t tmp1 = list[0].as_string();
			string_t tmp2 = list[1].as_string();
			std::string s_tmp1(tmp1.begin(), tmp1.end());
			std::string s_tmp2(tmp2.begin(), tmp2.end());
			value.push_back(std::pair<string, string>(s_tmp1, s_tmp2));
		}
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	return value;
}

void AdminHandler::parsing_error(http_request &request) {
	_log_access->error() << request.absolute_uri().to_string() + L" 400";
	json::value message;
	message[MESSAGE] = json::value::string(L"error parsing the input!");
	request.reply(status_codes::BadRequest, message);
}

AdminHandler::~AdminHandler() {}