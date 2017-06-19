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
class logManager;

class AdminHandler {
public:
	AdminHandler();
	AdminHandler(logManager *log_access);

	virtual void handle_create(World *world, vector<string_t> &paths, http_request &request) = 0;
	virtual void handle_update(World *world, vector<string_t> &paths, http_request &request) = 0;
	virtual void handle_read(World *world, vector<string_t> &paths, http_request &request) = 0;
	virtual void handle_delete(World *world, vector<string_t> &paths, http_request &request) = 0;

	bool check_field(json::value &data, string_t &s, http_request &request);
	void parsing_error(http_request &request);

	virtual ~AdminHandler();

protected:
	string_t NAME, UUID;
	string_t UMA_AGENT, UMA_SNAPSHOT, UMA_SENSOR;
	string_t UMA_AGENT_ID, UMA_SNAPSHOT_ID, UMA_SENSOR_ID;
	logManager *_log_access;
};

#endif