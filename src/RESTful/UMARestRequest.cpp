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
	string_t decodedUri = uri::decode(request.request_uri().query());
	_query = uri::split_query(decodedUri);
}

UMARestRequest::UMARestRequest(const web::uri &u, const http::method m) {
	_request = http_request(m);
	_request.set_request_uri(u);

	_body[U("data")] = json::value();
	_request.set_body(_body);
}

UMARestRequest::~UMARestRequest() {}


string UMARestRequest::getStringData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		string_t value = _data[nameT].as_string();
		return RestUtil::string_t2string(value);
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
	//convent from string_t to string
}


string UMARestRequest::getStringQuery(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_query, nameT);

	try {
		string_t value = web::uri::decode(_query[nameT]);
		return RestUtil::string_t2string(value);
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
	//convent from string_t to string
}


int UMARestRequest::getIntData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		int value = _data[nameT].as_integer();
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

int UMARestRequest::getIntQuery(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_query, nameT);
	try {
		int value = stoi(_query[nameT]);
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

double UMARestRequest::getDoubleData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		double value = _data[nameT].as_double();
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

double UMARestRequest::getDoubleQuery(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_query, nameT);
	try {
		double value = stod(_query[nameT]);
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

bool UMARestRequest::getBoolData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		bool value = _data[nameT].as_bool();
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

bool UMARestRequest::getBoolQuery(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_query, nameT);
	try {
		bool value = RestUtil::string_t2bool(_query[nameT]);
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<int> UMARestRequest::getInt1dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<int> value;
		json::array &list = _data[nameT].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_integer());
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<bool> UMARestRequest::getBool1dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<bool> value;
		json::array &list = _data[nameT].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_bool());
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<double> UMARestRequest::getDouble1dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<double> value;
		json::array &list = _data[nameT].as_array();
		for (int i = 0; i < list.size(); ++i) {
			value.push_back(list[i].as_double());
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<string> UMARestRequest::getString1dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<string> value;
		json::array &list = _data[nameT].as_array();
		for (int i = 0; i < list.size(); ++i) {
			string s = RestUtil::string_t2string(list[i].as_string());
			value.push_back(s);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<vector<int> > UMARestRequest::getInt2dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<vector<int> > value;
		auto &lists = _data[nameT].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<int> tmpValue;
			for (int j = 0; j < list.size(); ++j) {
				tmpValue.push_back(list[j].as_integer());
			}
			value.push_back(tmpValue);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<vector<bool> > UMARestRequest::getBool2dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<vector<bool> > value;
		auto &lists = _data[nameT].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<bool> tmpValue;
			for (int j = 0; j < list.size(); ++j) {
				tmpValue.push_back(list[j].as_bool());
			}
			value.push_back(tmpValue);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<vector<double> > UMARestRequest::getDouble2dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<vector<double> > value;
		auto &lists = _data[nameT].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<double> tmpValue;
			for (int j = 0; j < list.size(); ++j) {
				tmpValue.push_back(list[j].as_double());
			}
			value.push_back(tmpValue);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

vector<vector<string> > UMARestRequest::getString2dData(const string &name) {
	string_t nameT = RestUtil::string2string_t(name);
	RestUtil::checkField(_data, nameT);
	try {
		vector<vector<string> > value;
		auto &lists = _data[nameT].as_array();
		for (int i = 0; i < lists.size(); ++i) {
			auto &list = lists[i].as_array();
			vector<string> tmpValue;
			for (int j = 0; j < list.size(); ++j) {
				tmpValue.push_back(RestUtil::string_t2string(list[j].as_string()));
			}
			value.push_back(tmpValue);
		}
		return value;
	}
	catch (exception &e) {
		throw UMAInvalidArgsException("Cannot parsing the field " + name, false, &serverLogger);
	}
}

const string UMARestRequest::getRequestUrl() const{
	string_t url = _request.request_uri().path();
	return RestUtil::string_t2string(url);
}

const string UMARestRequest::getAbsoluteUrl() const{
	string_t url = _request.absolute_uri().path();
	return RestUtil::string_t2string(url);
}

void UMARestRequest::setMessage(const string message) {
	string_t s = RestUtil::string2string_t(message);
	_body[U("message")] = json::value::string(s);
}

void UMARestRequest::setStatusCode(const status_code statusCode) {
	_response.set_status_code(statusCode);
}

void UMARestRequest::setData(const string &name, int value) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = json::value::number(value);
}

void UMARestRequest::setData(const string &name, double value) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = json::value::number(value);
}

void UMARestRequest::setData(const string &name, bool value) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = json::value::boolean(value);
}

void UMARestRequest::setData(const string &name, string value) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = json::value::string(RestUtil::string2string_t(value));
}

void UMARestRequest::setData(const string &name, const std::vector<int> &list) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = RestUtil::vectorInt2Json(list);
}

void UMARestRequest::setData(const string &name, const std::vector<bool> &list) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = RestUtil::vectorBool2Json(list);
}

void UMARestRequest::setData(const string &name, const std::vector<double> &list) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = RestUtil::vectorDouble2Json(list);
}

void UMARestRequest::setData(const string &name, const std::vector<string> &list) {
	string_t nameT = RestUtil::string2string_t(name);
	_body[U("data")][nameT] = RestUtil::vectorString2Json(list);
}

void UMARestRequest::setData(const string &name, const std::vector<vector<int>> &lists) {
	string_t nameT = RestUtil::string2string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vectorInt2Json(lists[i]));
	}
	_body[U("data")][nameT] = json::value::array(results);
}

void UMARestRequest::setData(const string &name, const std::vector<vector<double>> &lists) {
	string_t nameT = RestUtil::string2string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vectorDouble2Json(lists[i]));
	}
	_body[U("data")][nameT] = json::value::array(results);
}

void UMARestRequest::setData(const string &name, const std::vector<vector<bool>> &lists) {
	string_t nameT = RestUtil::string2string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vectorBool2Json(lists[i]));
	}
	_body[U("data")][nameT] = json::value::array(results);
}

void UMARestRequest::setData(const string &name, const std::vector<vector<string>> &lists) {
	string_t nameT = RestUtil::string2string_t(name);
	vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		results.push_back(RestUtil::vectorString2Json(lists[i]));
	}
	_body[U("data")][nameT] = json::value::array(results);
}

void UMARestRequest::setData(const string &name, const std::map<string, string> &map) {
	string_t nameT = RestUtil::string2string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t keyT = RestUtil::string2string_t(it->first);
		string_t valueT = RestUtil::string2string_t(it->second);
		result[keyT] = json::value::string(valueT);
	}
	_body[U("data")][nameT] = result;
}

void UMARestRequest::setData(const string &name, const std::map<string, int> &map) {
	string_t nameT = RestUtil::string2string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t keyT = RestUtil::string2string_t(it->first);
		result[keyT] = json::value::number(it->second);
	}
	_body[U("data")][nameT] = result;
}

void UMARestRequest::setData(const string &name, const std::map<string, double> &map) {
	string_t nameT = RestUtil::string2string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t keyT = RestUtil::string2string_t(it->first);
		result[keyT] = json::value::number(it->second);
	}
	_body[U("data")][nameT] = result;
}

void UMARestRequest::setData(const string &name, const std::map<string, bool> &map) {
	string_t nameT = RestUtil::string2string_t(name);
	json::value result;
	for (auto it = map.begin(); it != map.end(); ++it) {
		string_t keyT = RestUtil::string2string_t(it->first);
		string_t valueT = RestUtil::string2string_t(to_string(it->second));
		result[keyT] = json::value::boolean(it->second);
	}
	_body[U("data")][nameT] = result;
}

bool UMARestRequest::checkDataField(const string &name) {
	const string_t nameT = RestUtil::string2string_t(name);
	return RestUtil::checkField(_data, nameT, false);
}

bool UMARestRequest::checkQueryField(const string &name) {
	const string_t nameT = RestUtil::string2string_t(name);
	return RestUtil::checkField(_query, nameT, false);
}

void UMARestRequest::reply() {
	_response.set_body(_body);
	pplx::task<void> replyTask = _request.reply(_response);
	replyTask.then([&]() {
		//do something after reply is succeed
	});
}