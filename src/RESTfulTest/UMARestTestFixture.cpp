#include "UMARestTestFixture.h"
#include "RestUtil.h"
#include "UMAException.h"

UMARestListener *UMARestTestFixture::listener = NULL;
UMARestRequest *UMARestTestFixture::request = NULL;
UMARestHandler *UMARestTestFixture::handler = NULL;
UMARestClient *UMARestTestFixture::client = NULL;

json::value UMARestTestFixture::clientData = json::value();
web::uri_builder UMARestTestFixture::clientQuery;

string UMARestTestFixture::serverString = "";
string UMARestTestFixture::clientString = "";
int UMARestTestFixture::serverInt = 0;
int UMARestTestFixture::clientInt = 0;
double UMARestTestFixture::serverDouble = 0;
double UMARestTestFixture::clientDouble = 0;
bool UMARestTestFixture::serverBool = false;
bool UMARestTestFixture::clientBool = false;
status_code UMARestTestFixture::serverStatusCode;
status_code UMARestTestFixture::clientStatusCode;

vector<int> UMARestTestFixture::serverInt1d;
vector<int> UMARestTestFixture::clientInt1d;
vector<bool> UMARestTestFixture::serverBool1d;
vector<bool> UMARestTestFixture::clientBool1d;
vector<string> UMARestTestFixture::serverString1d;
vector<string> UMARestTestFixture::clientString1d;
vector<double> UMARestTestFixture::serverDouble1d;
vector<double> UMARestTestFixture::clientDouble1d;
vector<vector<int>> UMARestTestFixture::serverInt2d;
vector<vector<int>> UMARestTestFixture::clientInt2d;
vector<vector<bool>> UMARestTestFixture::serverBool2d;
vector<vector<bool>> UMARestTestFixture::clientBool2d;
vector<vector<string>> UMARestTestFixture::serverString2d;
vector<vector<string>> UMARestTestFixture::clientString2d;
vector<vector<double>> UMARestTestFixture::serverDouble2d;
vector<vector<double>> UMARestTestFixture::clientDouble2d;
std::map<string, string> UMARestTestFixture::serverMapStringString;
std::map<string, string> UMARestTestFixture::clientMapStringString;
std::map<string, int> UMARestTestFixture::serverMapStringInt;
std::map<string, int> UMARestTestFixture::clientMapStringInt;
std::map<string, double> UMARestTestFixture::serverMapStringDouble;
std::map<string, double> UMARestTestFixture::clientMapStringDouble;
std::map<string, bool> UMARestTestFixture::serverMapStringBool;
std::map<string, bool> UMARestTestFixture::clientMapStringBool;

UMARestTestFixture::UMARestTestFixture() {
	if (!listener) {
		const string host = "localhost";
		const string port = "8001";
		const string url = "http://" + host + ":" + port;
		listener = new UMARestListener(url);
	}
	if (!handler) {
		handler = new UMARestTestHandler();
	}
	if (!client) {
		const string host = "localhost";
		const string port = "8001";
		const string url = "http://" + host + ":" + port + "/UMA/test";
		client = new UMARestClient(url);
	}
	listener->register_handler(handler);
	listener->add_path_to_handler("/UMA/test", "test_handler");
}

UMARestTestFixture::~UMARestTestFixture() {
	delete listener;
	delete handler;
	delete client;
	listener = NULL;
	handler = NULL;
	client = NULL;
}

void UMARestTestFixture::serverAction(string actionName){
	listener->listen();
}

void UMARestTestFixture::clientPostAction(string actionName){
	http_client client(U("http://localhost:8001/UMA/test"));
	http_request request(methods::POST);
	request.set_body(UMARestTestFixture::clientData);
	client.request(request).then([](http_response response) {
		//cout << "Get String Data Test is Done" << endl;
	}).wait();
}

void UMARestTestFixture::clientGetAction(string actionName) {
	http_client client(U("http://localhost:8001/UMA/test"));
	web::uri client_uri = clientQuery.to_uri();
	http_request request(methods::GET);
	request.set_request_uri(client_uri);
	client.request(request).then([](http_response response) {
		//cout << "Get String Data Test is Done" << endl;
		//wcout << response.to_string() << endl;
	}).wait();
}

void UMARestTestFixture::clientReceivingAction(string actionName) {
	http_client client(U("http://localhost:8001/UMA/test"));
	http_request request(methods::POST);
	request.set_body(UMARestTestFixture::clientData);
	client.request(request).then([](http_response response) {
		return response.extract_json();
	})
	.then([&](json::value d) {
		string type = RestUtil::string_t_to_string(clientData[U("type")].as_string());
		json::value data = d[U("data")];
		if (type == "set_message") {
			clientString = RestUtil::string_t_to_string(d[U("message")].as_string());
		}
		else if (type == "set_data_int") {
			clientInt = data[U("int")].as_integer();
		}
		else if (type == "set_data_double") {
			clientDouble = data[U("double")].as_double();
		}
		else if (type == "set_data_bool") {
			clientBool = data[U("bool")].as_bool();
		}
		else if (type == "set_data_string") {
			clientString = RestUtil::string_t_to_string(data[U("string")].as_string());
		}
		else if (type == "set_data_int1d") {
			vector<int> value;
			json::array &list = data[U("int1d")].as_array();
			for (int i = 0; i < list.size(); ++i) {
				value.push_back(list[i].as_integer());
			}
			clientInt1d = value;
		}
		else if (type == "set_data_string1d") {
			vector<string> value;
			json::array &list = data[U("string1d")].as_array();
			for (int i = 0; i < list.size(); ++i) {
				value.push_back(RestUtil::string_t_to_string(list[i].as_string()));
			}
			clientString1d = value;
		}
		else if (type == "set_data_double1d") {
			vector<double> value;
			json::array &list = data[U("double1d")].as_array();
			for (int i = 0; i < list.size(); ++i) {
				value.push_back(list[i].as_double());
			}
			clientDouble1d = value;
		}
		else if (type == "set_data_bool1d") {
			vector<bool> value;
			json::array &list = data[U("bool1d")].as_array();
			for (int i = 0; i < list.size(); ++i) {
				value.push_back(list[i].as_bool());
			}
			clientBool1d = value;
		}
		else if (type == "set_data_int2d") {
			vector<vector<int> > value;
			auto &lists = data[U("int2d")].as_array();
			for (int i = 0; i < lists.size(); ++i) {
				auto &list = lists[i].as_array();
				vector<int> tmp_value;
				for (int j = 0; j < list.size(); ++j) {
					tmp_value.push_back(list[j].as_integer());
				}
				value.push_back(tmp_value);
			}
			clientInt2d = value;
		}
		else if (type == "set_data_bool2d") {
			vector<vector<bool> > value;
			auto &lists = data[U("bool2d")].as_array();
			for (int i = 0; i < lists.size(); ++i) {
				auto &list = lists[i].as_array();
				vector<bool> tmp_value;
				for (int j = 0; j < list.size(); ++j) {
					tmp_value.push_back(list[j].as_bool());
				}
				value.push_back(tmp_value);
			}
			clientBool2d = value;
		}
		else if (type == "set_data_string2d") {
			vector<vector<string> > value;
			auto &lists = data[U("string2d")].as_array();
			for (int i = 0; i < lists.size(); ++i) {
				auto &list = lists[i].as_array();
				vector<string> tmp_value;
				for (int j = 0; j < list.size(); ++j) {
					tmp_value.push_back(RestUtil::string_t_to_string(list[j].as_string()));
				}
				value.push_back(tmp_value);
			}
			clientString2d = value;
		}
		else if (type == "set_data_double2d") {
			vector<vector<double> > value;
			auto &lists = data[U("double2d")].as_array();
			for (int i = 0; i < lists.size(); ++i) {
				auto &list = lists[i].as_array();
				vector<double> tmp_value;
				for (int j = 0; j < list.size(); ++j) {
					tmp_value.push_back(list[j].as_double());
				}
				value.push_back(tmp_value);
			}
			clientDouble2d = value;
		}
		else if (type == "set_data_map_string_string") {
			json::object clientMapStringStringObj = data[U("map_string_string")].as_object();
			for (auto it = clientMapStringStringObj.begin(); it != clientMapStringStringObj.end(); ++it) {
				clientMapStringString[RestUtil::string_t_to_string(it->first)] = RestUtil::string_t_to_string(it->second.as_string());
			}
		}
		else if (type == "set_data_map_string_int") {
			json::object clientMapStringIntObj = data[U("map_string_int")].as_object();
			for (auto it = clientMapStringIntObj.begin(); it != clientMapStringIntObj.end(); ++it) {
				clientMapStringInt[RestUtil::string_t_to_string(it->first)] = it->second.as_integer();
			}
		}
		else if (type == "set_data_map_string_double") {
			json::object clientMapStringDoubleObj = data[U("map_string_double")].as_object();
			for (auto it = clientMapStringDoubleObj.begin(); it != clientMapStringDoubleObj.end(); ++it) {
				clientMapStringDouble[RestUtil::string_t_to_string(it->first)] = it->second.as_double();
			}
		}
		else if (type == "set_data_map_string_bool") {
			json::object clientMapStringBoolObj = data[U("map_string_bool")].as_object();
			for (auto it = clientMapStringBoolObj.begin(); it != clientMapStringBoolObj.end(); ++it) {
				clientMapStringBool[RestUtil::string_t_to_string(it->first)] = it->second.as_bool();
			}
		}
	}).wait();
}

void UMARestTestFixture::testAction(string actionType) {
	thread t1(serverAction, "server");
	thread t2;
	if (actionType == "Post")
		t2 = thread(clientPostAction, "clientPost");
	else if (actionType == "Get")
		t2 = thread(clientGetAction, "clientGet");
	else if (actionType == "Receiving")
		t2 = thread(clientReceivingAction, "clientReceiving");
	t1.join();
	t2.join();
}

UMARestTestHandler::UMARestTestHandler(): UMARestHandler("test_handler") {}

UMARestTestHandler::~UMARestTestHandler() {}

void UMARestTestHandler::handle_create(UMARestRequest &request) {
	const string type = request.get_string_data("type");
	const string p_name = request.get_string_data("p_name");

	if (type == "get_string_data") {
		string s = request.get_string_data(p_name);
		UMARestTestFixture::serverString = s;
	}
	else if (type == "get_int_data") {
		int i = request.get_int_data(p_name);
		UMARestTestFixture::serverInt = i;
	}
	else if (type == "get_double_data") {
		double d = request.get_double_data(p_name);
		UMARestTestFixture::serverDouble = d;
	}
	else if (type == "get_bool_data") {
		bool b = request.get_bool_data(p_name);
		UMARestTestFixture::serverBool = b;
	}
	else if (type == "get_int1d_data") {
		vector<int> i1d = request.get_int1d_data(p_name);
		UMARestTestFixture::serverInt1d = i1d;
	}
	else if (type == "get_bool1d_data") {
		vector<bool> b1d = request.get_bool1d_data(p_name);
		UMARestTestFixture::serverBool1d = b1d;
	}
	else if (type == "get_string1d_data") {
		vector<string> s1d = request.get_string1d_data(p_name);
		UMARestTestFixture::serverString1d = s1d;
	}
	else if (type == "get_double1d_data") {
		vector<double> d1d = request.get_double1d_data(p_name);
		UMARestTestFixture::serverDouble1d = d1d;
	}
	else if (type == "get_int2d_data") {
		vector<vector<int>> i2d = request.get_int2d_data(p_name);
		UMARestTestFixture::serverInt2d = i2d;
	}
	else if (type == "get_bool2d_data") {
		vector<vector<bool>> b2d = request.get_bool2d_data(p_name);
		UMARestTestFixture::serverBool2d = b2d;
	}
	else if (type == "get_string2d_data") {
		vector<vector<string>> s2d = request.get_string2d_data(p_name);
		UMARestTestFixture::serverString2d = s2d;
	}
	else if (type == "get_double2d_data") {
		vector<vector<double>> d2d = request.get_double2d_data(p_name);
		UMARestTestFixture::serverDouble2d = d2d;
	}
	else if (type == "set_message") {
		request.set_message(UMARestTestFixture::serverString);
	}
	else if (type == "set_data_int") {
		request.set_data("int", UMARestTestFixture::serverInt);
	}
	else if (type == "set_data_double") {
		request.set_data("double", UMARestTestFixture::serverDouble);
	}
	else if (type == "set_data_bool") {
		request.set_data("bool", UMARestTestFixture::serverBool);
	}
	else if (type == "set_data_string") {
		request.set_data("string", UMARestTestFixture::serverString);
	}
	else if (type == "set_data_int1d") {
		request.set_data("int1d", UMARestTestFixture::serverInt1d);
	}
	else if (type == "set_data_double1d") {
		request.set_data("double1d", UMARestTestFixture::serverDouble1d);
	}
	else if (type == "set_data_string1d") {
		request.set_data("string1d", UMARestTestFixture::serverString1d);
	}
	else if (type == "set_data_bool1d") {
		request.set_data("bool1d", UMARestTestFixture::serverBool1d);
	}
	else if (type == "set_data_int2d") {
		request.set_data("int2d", UMARestTestFixture::serverInt2d);
	}
	else if (type == "set_data_double2d") {
		request.set_data("double2d", UMARestTestFixture::serverDouble2d);
	}
	else if (type == "set_data_string2d") {
		request.set_data("string2d", UMARestTestFixture::serverString2d);
	}
	else if (type == "set_data_bool2d") {
		request.set_data("bool2d", UMARestTestFixture::serverBool2d);
	}
	else if (type == "set_data_map_string_string") {
		request.set_data("map_string_string", UMARestTestFixture::serverMapStringString);
	}
	else if (type == "set_data_map_string_int") {
		request.set_data("map_string_int", UMARestTestFixture::serverMapStringInt);
	}
	else if (type == "set_data_map_string_double") {
		request.set_data("map_string_double", UMARestTestFixture::serverMapStringDouble);
	}
	else if (type == "set_data_map_string_bool") {
		request.set_data("map_string_bool", UMARestTestFixture::serverMapStringBool);
	}
	else {
		throw UMAException("cannot find any matching test case", UMAException::ERROR_LEVEL::ERROR, UMAException::CLIENT_DATA);
	}
}

void UMARestTestHandler::handle_read(UMARestRequest &request) {
	const string type = request.get_string_query("type");
	const string p_name = request.get_string_query("p_name");

	if (type == "get_string_query") {
		string s = request.get_string_query(p_name);
		UMARestTestFixture::serverString = s;
	}
	else if (type == "get_int_query") {
		int i = request.get_int_query(p_name);
		UMARestTestFixture::serverInt = i;
	}
	else if (type == "get_double_query") {
		double d = request.get_double_query(p_name);
		UMARestTestFixture::serverDouble = d;
	}
	else if (type == "get_bool_query") {
		bool b = request.get_bool_query(p_name);
		UMARestTestFixture::serverBool = b;
	}
	else {
		throw UMAException("cannot find any matching test case", UMAException::ERROR_LEVEL::ERROR, UMAException::CLIENT_DATA);
	}
}
void UMARestTestHandler::handle_update(UMARestRequest &request) {}
void UMARestTestHandler::handle_delete(UMARestRequest &request) {}