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
#include "logManager.h"
#include "UMAException.h"
#include <cpprest/json.h>

extern std::map<string, int> log_level;

listener::listener(const http::uri& url, std::map<string_t, vector<string_t>> &rest_map) : m_listener(http_listener(url)){
	//init the UMAC_access and UMAC log
	_log_path = "log";
	_log_access = new logManager(log_level["Server"], _log_path, "UMAC_access.txt", "listener");
	_log_server = new logManager(log_level["Server"], _log_path, "UMA_server.txt", "listener");

	_log_server->info() << U("Listening on the url ") + url.to_string();

	// every test will only have a unique world object
	_world = new World();
	_log_server->info() << "A new world is created";
	// support CRUD operation
	m_listener.support(methods::GET, std::bind(&listener::handle_get, this, std::placeholders::_1));
	_log_server->info() << "Init Get request success";

	m_listener.support(methods::PUT, std::bind(&listener::handle_put, this, std::placeholders::_1));
	_log_server->info() << "Init Put request success";

	m_listener.support(methods::POST, std::bind(&listener::handle_post, this, std::placeholders::_1));
	_log_server->info() << "Init Post request success";

	m_listener.support(methods::DEL, std::bind(&listener::handle_delete, this, std::placeholders::_1));
	_log_server->info() << "Init Delete request success";
	//create data handler

	try {
		init_restmap(rest_map);
		register_handler_factory();
	}
	catch (ServerException &e) {
		_log_server->error() << e.getErrorMessage();
		exit(0);
	}
}

void listener::register_handler_factory() {
	_world_handler = new WorldHandler("world", _log_access);
	_handler_factory[U("world")] = _world_handler;
	_log_server->info() << "A world handler is created";

	_agent_handler = new AgentHandler("agent", _log_access);
	_handler_factory[U("agent")] = _agent_handler;
	_log_server->info() << "An agent handler is created";

	_snapshot_handler = new SnapshotHandler("snapshot", _log_access);
	_handler_factory[U("snapshot")] = _snapshot_handler;
	_log_server->info() << "A snapshot handler is created";

	_data_handler = new DataHandler("data", _log_access);
	_handler_factory[U("data")] = _data_handler;
	_log_server->info() << "A data handler is created";

	_sensor_handler = new SensorHandler("sensor", _log_access);
	_handler_factory[U("sensor")] = _sensor_handler;
	_log_server->info() << "A sensor handler is created";

	_measurable_handler = new MeasurableHandler("measurable", _log_access);
	_handler_factory[U("measurable")] = _measurable_handler;
	_log_server->info() << "A measurable handler is created";

	_simulation_handler = new SimulationHandler("simulation", _log_access);
	_handler_factory[U("simulation")] = _simulation_handler;
	_log_server->info() << "A simulation handler is created";

	_matrix_handler = new MatrixHandler("matrix", _log_access);
	_handler_factory[U("matrix")] = _matrix_handler;
	_log_server->info() << "A matrix handler is created";
}


void listener::handle_get(http_request &request){
	string_t path = request.request_uri().path();
	AdminHandler *handler = find_handler(request);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << U("GET ") + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	//handler->REQUEST_MODE = handler->POST;
	http_response response;
	try {
		handler->handle_read(_world, path, request, response);
		status_code code = response.status_code();
		_log_access->info() << U("GET ") + request.absolute_uri().to_string() + status_code_to_string_t(code);
		request.reply(response);
	}
	catch (ClientException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("GET ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (ServerException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("GET ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (CoreException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("GET ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);

		if (e.getErrorLevel() == CoreException::CORE_FATAL) {
			_log_server->error() << "Shutting down server due to error: " + string_t_to_string(e.getErrorMessage());
			exit(0);
		}
	}
}

void listener::handle_put(http_request &request) {
	string_t path = request.request_uri().path();
	AdminHandler *handler = find_handler(request);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << U("POST") + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	//handler->REQUEST_MODE = handler->POST;
	http_response response;
	try {
		handler->handle_update(_world, path, request, response);
		status_code code = response.status_code();
		_log_access->info() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(code);
		request.reply(response);
	}
	catch (ClientException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (ServerException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (CoreException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);

		if (e.getErrorLevel() == CoreException::CORE_FATAL) {
			_log_server->error() << "Shutting down server due to error: " + string_t_to_string(e.getErrorMessage());
			exit(0);
		}
	}
}

void listener::handle_post(http_request &request){
	string_t path = request.request_uri().path();
	AdminHandler *handler = find_handler(request);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << U("POST") + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	//handler->REQUEST_MODE = handler->POST;
	http_response response;
	try {
		handler->handle_create(_world, path, request, response);
		status_code code = response.status_code();
		_log_access->info()<< U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(code);
		request.reply(response);
	}
	catch (ClientException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")]= json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (ServerException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (CoreException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("POST ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);

		if (e.getErrorLevel() == CoreException::CORE_FATAL) {
			_log_server->error() << "Shutting down server due to error: " + string_t_to_string(e.getErrorMessage());
			exit(0);
		}
	}
}

void listener::handle_delete(http_request &request){
	string_t path = request.request_uri().path();
	AdminHandler *handler = find_handler(request);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << U("DELETE ") + request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	//handler->REQUEST_MODE = handler->POST;
	http_response response;
	try {
		handler->handle_delete(_world, path, request, response);
		status_code code = response.status_code();
		_log_access->info() << U("DELETE ") + request.absolute_uri().to_string() + status_code_to_string_t(code);
		request.reply(response);
	}
	catch (ClientException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("DELETE ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (ServerException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("DELETE ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);
	}
	catch (CoreException &e) {
		status_code error_code = e.getErrorCode();
		_log_access->error() << U("DELETE ") + request.absolute_uri().to_string() + status_code_to_string_t(error_code);
		response.set_status_code(error_code);
		json::value message;
		message[U("message")] = json::value(e.getErrorMessage());
		response.set_body(message);
		request.reply(response);

		if (e.getErrorLevel() == CoreException::CORE_FATAL) {
			_log_server->error() << "Shutting down server due to error: " + string_t_to_string(e.getErrorMessage());
			exit(0);
		}
	}
}

void listener::init_restmap(std::map < string_t, vector<string_t>> &rest_map) {
	try {
		for (auto it = rest_map.begin(); it != rest_map.end(); ++it) {
			string_t s_factory = it->first;
			for (int i = 0; i < it->second.size(); ++i) {
				string_t s_path = it->second[i];
				_path_to_handler[s_path] = s_factory;
				_log_server->info() << "add mapping from " + string_t_to_string(s_path) + " to \"" + string_t_to_string(s_factory) + "\"";
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