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
class WorldHandler;
class AgentHandler;
class SnapshotHandler;
class DataHandler;
class SensorHandler;
class MeasurableHandler;
class SimulationHandler;
class MatrixHandler;
class logging;
class logManager;

class listener
{
public:
	listener(string url);
	http_listener m_listener;

protected:
	World *_world;
	WorldHandler *_world_handler;
	AgentHandler *_agent_handler;
	SnapshotHandler *_snapshot_handler;
	DataHandler *_data_handler;
	SensorHandler *_sensor_handler;
	MeasurableHandler *_measurable_handler;
	SimulationHandler *_simulation_handler;
	MatrixHandler *_matrix_handler;
	std::map<string_t, AdminHandler*> _handler_factory;
	std::map<string_t, string_t> _path_to_handler;

private:
	void init_restmap(std::map < string_t, vector<string_t>> &rest_map);
	void register_handler_factory();
	void handle_get(http_request &request);
	void handle_put(http_request &request);
	void handle_post(http_request &request);
	void handle_delete(http_request &request);
	void handle(http_request &request, string request_type);
	AdminHandler *find_handler(http_request &request);
};

#endif