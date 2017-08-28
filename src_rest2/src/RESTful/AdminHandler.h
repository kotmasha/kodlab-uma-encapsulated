#ifndef _ADMINHANDLER_
#define _ADMINHANDLER_

#include "Global.h"
#include <cpprest/http_listener.h>
using namespace web::http::experimental::listener;
using namespace web::http;
using namespace web;
using namespace utility;
using namespace std;

class World;
class Agent;
class Snapshot;
class Sensor;
class logManager;

class AdminHandler {
public:
	AdminHandler(string handler_factor, logManager *log_access);

	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response) = 0;
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response) = 0;
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response) = 0;
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response) = 0;

	bool check_field(json::value &data, string_t &s, bool hard_check = true);
	bool check_field(map<string_t, string_t> &query, string_t &s, bool hard_check = true);
	void vector_int_to_array(std::vector<int> &list, std::vector<json::value> &json_list);
	void vector_bool_to_array(std::vector<bool> &list, std::vector<json::value> &json_list);
	void vector_string_to_array(std::vector<string> &list, std::vector<json::value> &json_list);
	Agent *get_agent_by_id(World *world, string agent_id, http_request &request, http_response &response);
	bool get_snapshot_by_id(Agent *agent, string snapshot_id, Snapshot *&snapshot, http_request &request);
	bool get_sensor_by_id(Snapshot *snapshot, string &sensor_id, Sensor *&sensor, http_request &request);

	string_t MESSAGE;
	string_t REQUEST_MODE;
	string_t GET, POST, PUT, DELETE;

	virtual ~AdminHandler();

protected:
	string_t UUID;
	string_t UMA_AGENT, UMA_SNAPSHOT, UMA_SENSOR, UMA_SENSOR_PAIR, UMA_MEASURABLE, UMA_MEASURABLE_PAIR;
	string_t UMA_AGENT_ID, UMA_SNAPSHOT_ID, UMA_SENSOR_ID, UMA_MEASURABLE_ID;
	string_t UMA_SENSORS;
	string_t DATA;
	logManager *_log_access;

	string _handler_factory;

protected:
	string get_string_input(json::value &data, string_t &name);
	string get_string_input(map<string_t, string_t> &query, string_t &name);
	int get_int_input(json::value &data, string_t &name);
	int get_int_input(map<string_t, string_t> &query, string_t &name, http_request &request);
	double get_double_input(json::value &data, string_t &name);
	double get_double_input(map<string_t, string_t> &query, string_t &name, http_request &request);
	bool get_bool_input(json::value &data, string_t &name);
	bool get_bool_input(map<string_t, string_t> &query, string_t &name, http_request &request);
	vector<int> get_int1d_input(json::value &data, string_t &name, http_request &request);
	vector<bool> get_bool1d_input(json::value &data, string_t &name);
	vector<vector<bool> > get_bool2d_input(json::value &data, string_t &name);
	vector<string> get_string1d_input(json::value &data, string_t &name);
	vector<std::pair<string, string> > get_string_pair1d_input(json::value &data, string_t &name);
};

#endif