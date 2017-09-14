#ifndef _LISTENER_
#define _LISTENER_

#include <cpprest/http_listener.h>
#include "Global.h"

using namespace web::http::experimental::listener;
using namespace web::http;
using namespace web;
using namespace utility;
using namespace std;

class World;
class AdminHandler;
class DataHandler;
class AgentHandler;
class SnapshotHandler;
class SensorHandler;
class MeasurableHandler;
class DataValidationHandler;
class SimulationHandler;
class MatrixHandler;
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
	AgentHandler *_agent_handler;
	SnapshotHandler *_snapshot_handler;
	SensorHandler *_sensor_handler;
	MeasurableHandler *_measurable_handler;
	DataValidationHandler *_data_validation_handler;
	SimulationHandler *_simulation_handler;
	MatrixHandler *_matrix_handler;
	string _log_path;
	logManager *_log_access, *_log_server;
	std::map<string_t, AdminHandler*> _handler_factory;
	std::map<string_t, string_t> _path_to_handler;

private:
	void init_restmap();
	void register_handler_factory();
	void handle_get(http_request &request);
	void handle_put(http_request &request);
	void handle_post(http_request &request);
	void handle_delete(http_request &request);
	AdminHandler *find_handler(http_request &request);
};

#endif