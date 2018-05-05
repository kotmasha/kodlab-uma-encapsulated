#include "UMARestRequest.h"
#include "RestUtil.h"
#include "UMAException.h"
#include "Logger.h"

static Logger accessLogger("Access", "log/UMAC_access.log");
static Logger serverLogger("Server", "log/UMA_server.log");

UMARestRequest::UMARestRequest(const http_request &request): _request(request) {
	_response = http_response();
	_body[U("data")] = json::value();
	_response.set_body(_body);
	//while (!request.extract_json().is_done()) continue;
	//_data = request.extract_json().get();

	//get the data, due to the data arriving is async, need to use task to get it
	request.extract_json().then([&](pplx::task<json::value> v){
		_data = v.get();
	}).wait();

	//get the query
	string_t decoded_uri = uri::decode(request.request_uri().query());
	_query = uri::split_query(decoded_uri);
}

UMARestRequest::UMARestRequest(const web::uri &u, const http::method m) {
	_request = http_request(m);
	_request.set_request_uri(u);

	_body[U("data")] = json::value();
	_request.set_body(_body);
}

UMARestRequest::~UMARestRequest() {}


string UMARestRequest::get_string_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		string_t value = _data[name_t].as_string();
		return RestUtil::string_t_to_string(value);
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
	//convent from string_t to string
}


string UMARestRequest::get_string_query(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_query, name_t);

	try {
		string_t value = web::uri::decode(_query[name_t]);
		return RestUtil::string_t_to_string(value);
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
	//convent from string_t to string
}


int UMARestRequest::get_int_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		int value = _data[name_t].as_integer();
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " +name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

int UMARestRequest::get_int_query(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_query, name_t);
	try {
		int value = stoi(_query[name_t]);
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

double UMARestRequest::get_double_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		double value = _data[name_t].as_double();
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

double UMARestRequest::get_double_query(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_query, name_t);
	try {
		double value = stod(_query[name_t]);
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

bool UMARestRequest::get_bool_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		bool value = _data[name_t].as_bool();
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

bool UMARestRequest::get_bool_query(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_query, name_t);
	try {
		bool value = RestUtil::string_t_to_bool(_query[name_t]);
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<int> UMARestRequest::get_int1d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<int> value;
		json::array &list = _data[name_t].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_integer());
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<bool> UMARestRequest::get_bool1d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<bool> value;
		json::array &list = _data[name_t].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_bool());
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<double> UMARestRequest::get_double1d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<double> value;
		json::array &list = _data[name_t].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_double());
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<string> UMARestRequest::get_string1d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<string> value;
		json::array &list = _data[name_t].as_array();
		for (int i = 0; i < list.size(); ++i) {
			string s = RestUtil::string_t_to_string(list[i].as_string());
			value.push_back(s);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<vector<int> > UMARestRequest::get_int2d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<vector<int> > value;
		auto &lists = _data[name_t].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<int> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(list[j].as_integer());
			}
			value.push_back(tmp_value);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<vector<bool> > UMARestRequest::get_bool2d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<vector<bool> > value;
		auto &lists = _data[name_t].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<bool> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(list[j].as_bool());
			}
			value.push_back(tmp_value);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<vector<double> > UMARestRequest::get_double2d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<vector<double> > value;
		auto &lists = _data[name_t].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<double> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(list[j].as_double());
			}
			value.push_back(tmp_value);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

vector<vector<string> > UMARestRequest::get_string2d_data(const string &name) {
	string_t name_t = RestUtil::string_to_string_t(name);
	RestUtil::check_field(_data, name_t);
	try {
		vector<vector<string> > value;
		auto &lists = _data[name_t].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<string> tmp_value;
			for (int j = 0; j < list.size(); ++j) {
				tmp_value.push_back(RestUtil::string_t_to_string(list[j].as_string()));
			}
			value.push_back(tmp_value);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAException("Cannot parsing the field " + name, UMAException::ERROR_LEVEL::ERROR, UMAException::ERROR_TYPE::CLIENT_DATA);
	}
}

const string UMARestRequest::get_request_url() const{
	string_t url = _request.request_uri().path();
	return RestUtil::string_t_to_string(url);
}

const string UMARestRequest::get_absolute_url() const{
	string_t url = _request.absolute_uri().path();
	return RestUtil::string_t_to_string(url);
}

void UMARestRequest::set_message(const string message) {
	string_t s = RestUtil::string_to_string_t(message);
	_body[U("message")] = json::value::string(s);
}

void UMARestRequest::set_status_code(const status_code status_code) {
	_response.set_status_code(status_code);
}

void UMARestRequest::set_data(const string &name, int value) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = json::value::number(value);
}

void UMARestRequest::set_data(const string &name, double value) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = json::value::number(value);
}

void UMARestRequest::set_data(const string &name, bool value) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = json::value::boolean(value);
}

void UMARestRequest::set_data(const string &name, string value) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = json::value::string(RestUtil::string_to_string_t(value));
}

void UMARestRequest::set_data(const string &name, const std::vector<int> &list) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = RestUtil::vector_int_to_json(list);
}

void UMARestRequest::set_data(const string &name, const std::vector<bool> &list) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = RestUtil::vector_bool_to_json(list);
}

void UMARestRequest::set_data(const string &name, const std::vector<double> &list) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = RestUtil::vector_double_to_json(list);
}

void UMARestRequest::set_data(const string &name, const std::vector<string> &list) {
	string_t name_t = RestUtil::string_to_string_t(name);
	_body[U("data")][name_t] = RestUtil::vector_string_to_json(list);
}

void UMARestRequest::set_data(const string &name, const std::vector<vector<int>> &lists) {
	string_t name_t = RestUtil::string_to_string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vector_int_to_json(lists[i]));
	}
	_body[U("data")][name_t] = json::value::array(results);
}

void UMARestRequest::set_data(const string &name, const std::vector<vector<double>> &lists) {
	string_t name_t = RestUtil::string_to_string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vector_double_to_json(lists[i]));
	}
	_body[U("data")][name_t] = json::value::array(results);
}

void UMARestRequest::set_data(const string &name, const std::vector<vector<bool>> &lists) {
	string_t name_t = RestUtil::string_to_string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vector_bool_to_json(lists[i]));
	}
	_body[U("data")][name_t] = json::value::array(results);
}

void UMARestRequest::set_data(const string &name, const std::vector<vector<string>> &lists) {
	string_t name_t = RestUtil::string_to_string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vector_string_to_json(lists[i]));
	}
	_body[U("data")][name_t] = json::value::array(results);
}

void UMARestRequest::set_data(const string &name, const std::map<string, string> &map) {
	string_t name_t = RestUtil::string_to_string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t key_t = RestUtil::string_to_string_t(it->first);
		string_t value_t = RestUtil::string_to_string_t(it->second);
		result[key_t] = json::value::string(value_t);
	}
	_body[U("data")][name_t] = result;
}

void UMARestRequest::set_data(const string &name, const std::map<string, int> &map) {
	string_t name_t = RestUtil::string_to_string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t key_t = RestUtil::string_to_string_t(it->first);
		result[key_t] = json::value::number(it->second);
	}
	_body[U("data")][name_t] = result;
}

void UMARestRequest::set_data(const string &name, const std::map<string, double> &map) {
	string_t name_t = RestUtil::string_to_string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t key_t = RestUtil::string_to_string_t(it->first);
		result[key_t] = json::value::number(it->second);
	}
	_body[U("data")][name_t] = result;
}

void UMARestRequest::set_data(const string &name, const std::map<string, bool> &map) {
	string_t name_t = RestUtil::string_to_string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t key_t = RestUtil::string_to_string_t(it->first);
		string_t value_t = RestUtil::string_to_string_t(to_string(it->second));
		result[key_t] = json::value::boolean(it->second);
	}
	_body[U("data")][name_t] = result;
}

bool UMARestRequest::check_data_field(const string &name) {
	const string_t name_t = RestUtil::string_to_string_t(name);
	return RestUtil::check_field(_data, name_t, false);
}

bool UMARestRequest::check_query_field(const string &name) {
	const string_t name_t = RestUtil::string_to_string_t(name);
	return RestUtil::check_field(_query, name_t, false);
}

void UMARestRequest::reply() {
	_response.set_body(_body);
	pplx::task<void> reply_task = _request.reply(_response);
	reply_task.then([&]() {
		//do something after reply is succeed
	});
}