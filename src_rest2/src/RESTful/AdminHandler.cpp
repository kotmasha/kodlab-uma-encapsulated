#include "AdminHandler.h"
#include "logManager.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "UMAException.h"

bool to_bool(string_t &s) {
	return s == U("true") || s == U("True");
}

AdminHandler::AdminHandler(string handler_factory):_handler_factory(handler_factory){
	UUID = U("uuid");
	UMA_AGENT = U("agent");
	UMA_SNAPSHOT = U("snapshot");
	UMA_SENSOR = U("sensor");
	UMA_SENSOR_PAIR = U("sensor_pair");
	UMA_MEASURABLE = U("measurable");
	UMA_MEASURABLE_PAIR = U("measurable_pair");
	UMA_AGENT_ID = U("agent_id");
	UMA_SNAPSHOT_ID = U("snapshot_id");
	UMA_SENSOR_ID = U("sensor_id");
	UMA_MEASURABLE_ID = U("measurable_id");

	UMA_SENSORS = U("sensors");
	UMA_SIGNAL = U("signal");
	UMA_SIGNALS = U("signals");
	UMA_LOAD = U("load");

	DATA = U("data");

	GET = U(" GET ");
	PUT = U(" PUT ");
	POST = U(" POST ");
	DELETE = U(" DELETE ");

	MESSAGE = U("message");
}

bool AdminHandler::check_field(json::value &data, string_t &s, bool hard_check) {
	if (!data.has_field(s)) {
		if (hard_check) {
			throw ClientException("Coming request is missing necessary fields", ClientException::CLIENT_ERROR, status_codes::BadRequest);
		}
		return false;
	}
	return true;
}


bool AdminHandler::check_field(map<string_t, string_t> &query, string_t &s, bool hard_check) {
	if (query.find(s) == query.end()) {
		if (hard_check) {
			throw ClientException("Coming request is missing necessary fields", ClientException::CLIENT_ERROR, status_codes::BadRequest);
		}
		return false;
	}
	return true;
}


void AdminHandler::vector_int_to_array(std::vector<int> &list, std::vector<json::value> &json_list) {
	for (int i = 0; i < list.size(); ++i) {
		json_list.push_back(json::value::number(list[i]));
	}
}

void AdminHandler::vector_double_to_array(std::vector<double> &list, std::vector<json::value> &json_list) {
	for (int i = 0; i < list.size(); ++i) {
		json_list.push_back(json::value::number(list[i]));
	}
}

void AdminHandler::vector_bool_to_array(std::vector<bool> &list, std::vector<json::value> &json_list) {
	for (int i = 0; i < list.size(); ++i) {
		json_list.push_back(json::value::boolean(list[i]));
	}
}

void AdminHandler::vector_bool2d_to_array(std::vector<vector<bool> > &lists, std::vector<json::value> &json_lists) {
	for (int i = 0; i < lists.size(); ++i) {
		vector<json::value> value;
		vector_bool_to_array(lists[i], value);
		json_lists.push_back(json::value::array(value));
	}
}

void AdminHandler::vector_int2d_to_array(std::vector<vector<int> > &lists, std::vector<json::value> &json_lists) {
	for (int i = 0; i < lists.size(); ++i) {
		vector<json::value> value;
		vector_int_to_array(lists[i], value);
		json_lists.push_back(json::value::array(value));
	}
}

void AdminHandler::vector_double2d_to_array(std::vector<vector<double> > &lists, std::vector<json::value> &json_lists) {
	for (int i = 0; i < lists.size(); ++i) {
		vector<json::value> value;
		vector_double_to_array(lists[i], value);
		json_lists.push_back(json::value::array(value));
	}
}

void AdminHandler::vector_string_to_array(std::vector<string> &list, std::vector<json::value> &json_list) {
	for (int i = 0; i < list.size(); ++i) {
		string_t tmp(list[i].begin(), list[i].end());
		json_list.push_back(json::value::string(tmp));
	}
}

Agent *AdminHandler::get_agent_by_id(World *world, string agent_id, http_request &request, http_response &response) {
	Agent *agent = world->getAgent(agent_id);
	if (agent == NULL) {
		throw ClientException("Cannot find the agent id!", ClientException::CLIENT_ERROR, status_codes::NotFound);
	}
	return agent;
}

string AdminHandler::get_string_input(json::value &data, string_t &name) {
	check_field(data, name);
	string_t value;
	try {
		value = data[name].as_string();
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	//convent from string_t to string
	return string_t_to_string(value);
}


string AdminHandler::get_string_input(map<string_t, string_t> &query, string_t &name) {
	check_field(query, name);
	string_t value;
	try {
		value = web::uri::decode(query[name]);
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	//convent from string_t to string
	return string_t_to_string(value);
}


int AdminHandler::get_int_input(json::value &data, string_t &name) {
	check_field(data, name);
	int value;
	try {
		value = data[name].as_integer();
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}

	return value;
}

/*
int AdminHandler::get_int_input(map<string_t, string_t> &query, string_t &name, http_request &request) {
	if (!check_field(query, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	int value;
	try {
		value = stoi(query[name]);
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	//convent from wstring to string
	return value;
}

*/

double AdminHandler::get_double_input(json::value &data, string_t &name) {
	check_field(data, name);
	double value;
	try {
		value = data[name].as_double();
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}

	return value;
}

/*

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

*/

bool AdminHandler::get_bool_input(json::value &data, string_t &name) {
	check_field(data, name);
	bool value;
	try {
		value = data[name].as_bool();
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}

	return value;
}

/*

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

vector<int> AdminHandler::get_int1d_input(json::value &data, string_t &name, http_request &request) {
	if (!check_field(data, name, request)) throw CLIENT_EXCEPTION::CLIENT_ERROR;
	vector<int> value;
	try {
		auto &list = data[name].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_integer());
		}
	}
	catch (exception &e) {
		cout << e.what() << endl;
		parsing_error(request);
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	return value;
}

*/

vector<bool> AdminHandler::get_bool1d_input(json::value &data, string_t &name) {
	check_field(data, name);
	vector<bool> value;
	try {
		auto &list = data[name].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_bool());
		}
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}

	return value;
}

vector<vector<bool> > AdminHandler::get_bool2d_input(json::value &data, string_t &name) {
	check_field(data, name);
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
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	return value;
}

vector<vector<double> > AdminHandler::get_double2d_input(json::value &data, string_t &name) {
	check_field(data, name);
	vector<vector<double> > value;
	try {
		auto &lists = data[name].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<double> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(list[j].as_double());
			}
			value.push_back(tmp_value);
		}
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	return value;
}

vector<vector<int> > AdminHandler::get_int2d_input(json::value &data, string_t &name) {
	check_field(data, name);
	vector<vector<int> > value;
	try {
		auto &lists = data[name].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<int> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(list[j].as_integer());
			}
			value.push_back(tmp_value);
		}
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	return value;
}

vector<string> AdminHandler::get_string1d_input(json::value &data, string_t &name) {
	check_field(data, name);
	vector<string> value;
	try {
		auto &list = data[name].as_array();
		for (int i = 0; i < list.size(); ++i) {
			string_t tmp = list[i].as_string();
			string s_tmp = string_t_to_string(tmp);
			value.push_back(s_tmp);
		}
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	return value;
}

vector<double> AdminHandler::get_double1d_input(json::value &data, string_t &name) {
	check_field(data, name);
	vector<double> value;
	try {
		auto &list = data[name].as_array();
		for (int i = 0; i < list.size(); ++i) {
			double tmp = list[i].as_double();
			value.push_back(tmp);
		}
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	return value;
}

vector<std::pair<string, string> > AdminHandler::get_string_pair1d_input(json::value &data, string_t &name) {
	check_field(data, name);
	vector<std::pair<string, string> > value;
	try {
		auto &lists = data[name].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto list = lists[i].as_array();
			if (list.size() != 2) throw ClientException();
			string_t tmp1 = list[0].as_string();
			string_t tmp2 = list[1].as_string();
			std::string s_tmp1(tmp1.begin(), tmp1.end());
			std::string s_tmp2(tmp2.begin(), tmp2.end());
			value.push_back(std::pair<string, string>(s_tmp1, s_tmp2));
		}
	}
	catch (exception &e) {
		throw ClientException("Cannot parsing the field " + string_t_to_string(name), ClientException::CLIENT_ERROR, status_codes::BadRequest);
	}
	return value;
}

AdminHandler::~AdminHandler() {}