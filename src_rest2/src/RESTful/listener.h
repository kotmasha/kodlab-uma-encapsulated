#ifndef _LISTENER_
#define _LISTENER_

#include <cpprest\http_listener.h>
#include "Global.h"

using namespace web::http::experimental::listener;
using namespace web::http;
using namespace web;
using namespace utility;
using namespace std;

class World;
class AdminHandler;
class DataHandler;
class ObjectHandler;
class DataValidationHandler;
class SimulationHandler;
class logging;
class logManager;

class listener
{
public:
	listener(const http::uri& url);
	http_listener m_listener;

protected:
	World *_world;
	DataHandler *_data_handler;
	ObjectHandler *_object_handler;
	DataValidationHandler *_data_validation_handler;
	SimulationHandler *_simulation_handler;
	string _log_path;
	logManager *_log_access, *_log_server;

private:
	void handle_get(http_request request);
	void handle_put(http_request request);
	void handle_post(http_request request);
	void handle_delete(http_request request);
	AdminHandler *find_handler(vector<string_t> &paths);
};

#endif