#include "listener.h"
#include "World.h"
#include "WorldHandler.h"
#include "AgentHandler.h"
#include "SimulationHandler.h"
#include "AdminHandler.h"
#include "SnapshotHandler.h"
#include "DataHandler.h"
#include "SensorHandler.h"
#include "MeasurableHandler.h"
#include "MatrixHandler.h"
#include "UMAException.h"
#include "Logger.h"
#include <cpprest/json.h>

extern Logger accessLogger;
extern Logger serverLogger;

listener::listener(const http::uri& url, std::map<string_t, vector<string_t>> &rest_map) : m_listener(http_listener(url)){
	//init the UMAC_access and UMAC log

	serverLogger.info("Listening on the url" + string_t_to_string(url.to_string()));

	// every test will only have a unique world object
	_world = new World();
	serverLogger.info("A new world is created");
	// support CRUD operation
	m_listener.support(methods::GET, std::bind(&listener::handle_get, this, std::placeholders::_1));
	serverLogger.info("Init Get request success");

	m_listener.support(methods::PUT, std::bind(&listener::handle_put, this, std::placeholders::_1));
	serverLogger.info("Init Put request success");

	m_listener.support(methods::POST, std::bind(&listener::handle_post, this, std::placeholders::_1));
	serverLogger.info("Init Post request success");

	m_listener.support(methods::DEL, std::bind(&listener::handle_delete, this, std::placeholders::_1));
	serverLogger.info("Init Delete request success");
	//create data handler

	try {
		init_restmap(rest_map);
		register_handler_factory();
	}
	catch (ServerException &e) {
		serverLogger.error(e.getErrorMessage());
		exit(0);
	}
}

void listener::register_handler_factory() {
	_world_handler = new WorldHandler("world");
	_handler_factory[U("world")] = _world_handler;
	serverLogger.info("A world handler is created");

	_agent_handler = new AgentHandler("agent");
	_handler_factory[U("agent")] = _agent_handler;
	serverLogger.info("An agent handler is created");

	_snapshot_handler = new SnapshotHandler("snapshot");
	_handler_factory[U("snapshot")] = _snapshot_handler;
	serverLogger.info("A snapshot handler is created");

	_data_handler = new DataHandler("data");
	_handler_factory[U("data")] = _data_handler;
	serverLogger.info("A data handler is created");

	_sensor_handler = new SensorHandler("sensor");
	_handler_factory[U("sensor")] = _sensor_handler;
	serverLogger.info("A sensor handler is created");

	_measurable_handler = new MeasurableHandler("measurable");
	_handler_factory[U("measurable")] = _measurable_handler;
	serverLogger.info("A measurable handler is created");

	_simulation_handler = new SimulationHandler("simulation");
	_handler_factory[U("simulation")] = _simulation_handler;
	serverLogger.info("A simulation handler is created");

	_matrix_handler = new MatrixHandler("matrix");
	_handler_factory[U("matrix")] = _matrix_handler;
	serverLogger.info("A matrix handler is created");
}

void listener::handle(http_request &request, string request_type) {
	string_t path = request.request_uri().path();
	AdminHandler *handler = find_handler(request);
	// if no handlers can be found
	if (handler == NULL) {
		accessLogger.error(request_type + " " + string_t_to_string(request.absolute_uri().to_string()) + " 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	//handler->REQUEST_MODE = handler->POST;
	http_response response;
	try {
		if(request_type == "GET") handler->handle_read(_world, path, request, response);
		else if(request_type == "POST") handler->handle_create(_world, path, request, response);
		else if(request_type == "PUT") handler->handle_update(_world, path, request, response);
		else handler->handle_delete(_world, path, request, response);
		status_code code = response.status_code();
		accessLogger.info(request_type + " " + string_t_to_string(request.absolute_uri().to_string()) + " " + string_t_to_string(status_code_to_string_t(code)));
		request.reply(response);
	}
	catch (ClientException &e) {
		status_code error_code = e.getErrorCode();
		accessLogger.error(request_type + " " + string_t_to_string(request.absolute_uri().to_string()) + " " + string_t_to_string(status_code_to_string_t(error_code)));
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(string_to_string_t(e.getErrorMessage()));
		response.set_body(message);
		request.reply(response);
	}
	catch (ServerException &e) {
		status_code error_code = e.getErrorCode();
		accessLogger.error("GET " + string_t_to_string(request.absolute_uri().to_string()) + " " + string_t_to_string(status_code_to_string_t(error_code)));
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(string_to_string_t(e.getErrorMessage()));
		response.set_body(message);
		request.reply(response);
	}
	catch (CoreException &e) {
		status_code error_code = e.getErrorCode();
		accessLogger.error(request_type + " " + string_t_to_string(request.absolute_uri().to_string()) + " " + string_t_to_string(status_code_to_string_t(error_code)));
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(string_to_string_t(e.getErrorMessage()));
		response.set_body(message);
		request.reply(response);

		if (e.getErrorLevel() == CoreException::CORE_FATAL) {
			accessLogger.error("Shutting down server due to error: " + e.getErrorMessage());
			exit(0);
		}
	}
}

void listener::handle_get(http_request &request){
	handle(request, "GET");
}

void listener::handle_put(http_request &request) {
	handle(request, "PUT");
}

void listener::handle_post(http_request &request){
	handle(request, "POST");
}

void listener::handle_delete(http_request &request){
	handle(request, "DELETE");
}

void listener::init_restmap(std::map < string_t, vector<string_t>> &rest_map) {
	try {
		for (auto it = rest_map.begin(); it != rest_map.end(); ++it) {
			string_t s_factory = it->first;
			for (int i = 0; i < it->second.size(); ++i) {
				string_t s_path = it->second[i];
				_path_to_handler[s_path] = s_factory;
				serverLogger.info("add mapping from " + string_t_to_string(s_path) + " to \"" + string_t_to_string(s_factory) + "\"");
			}
		}
	}
	catch (exception &e) {
		throw ServerException("Having some problem mapping restmap!", ServerException::SERVER_FATAL);
	}
}

AdminHandler *listener::find_handler(http_request &request) {
	string_t path = request.request_uri().path();
	try {
		string_t factory = _path_to_handler[path];
		AdminHandler *handler = _handler_factory[factory];
		return handler;
	}
	catch (UMAException &e) {
		return NULL;
	}
}