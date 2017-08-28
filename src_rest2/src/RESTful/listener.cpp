#include "listener.h"
#include "World.h"
#include "DataHandler.h"
#include "AgentHandler.h"
#include "SimulationHandler.h"
#include "AdminHandler.h"
#include "SnapshotHandler.h"
#include "SensorHandler.h"
#include "MeasurableHandler.h"
#include "logManager.h"
#include "UMAException.h"
#include <cpprest/json.h>

listener::listener(const http::uri& url) : m_listener(http_listener(url)){
	//init the UMAC_access and UMAC log
	_log_path = "log";
	_log_access = new logManager(logging::VERBOSE, _log_path, "UMAC_access.txt", typeid(*this).name());
	_log_server = new logManager(logging::VERBOSE, _log_path, "UMA_server.txt", typeid(*this).name());

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
		init_restmap();
		register_handler_factory();
	}
	catch (ServerException &e) {
		_log_server->error() << e.getErrorMessage();
		exit(0);
	}
}

void listener::register_handler_factory() {
	_data_handler = new DataHandler("data", _log_access);
	_handler_factory[U("data")] = _data_handler;
	_log_server->info() << "A data handler is created";

	_agent_handler = new AgentHandler("agent", _log_access);
	_handler_factory[U("agent")] = _agent_handler;
	_log_server->info() << "An agent handler is created";

	_snapshot_handler = new SnapshotHandler("snapshot", _log_access);
	_handler_factory[U("snapshot")] = _snapshot_handler;
	_log_server->info() << "A snapshot handler is created";

	_sensor_handler = new SensorHandler("sensor", _log_access);
	_handler_factory[U("sensor")] = _sensor_handler;
	_log_server->info() << "A sensor handler is created";

	_measurable_handler = new MeasurableHandler("measurable", _log_access);
	_handler_factory[U("measurable")] = _measurable_handler;
	_log_server->info() << "A measurable handler is created";

	//_data_validation_handler = new DataValidationHandler("validation", _log_access);
	//_handler_factory[U("validation")] = _data_validation_handler;
	//_log_server->info() << "A data validation handler is created";

	_simulation_handler = new SimulationHandler("simulation", _log_access);
	_handler_factory[U("simulation")] = _simulation_handler;
	_log_server->info() << "A simulation handler is created";
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

		if (e.getErrorLevel() == CoreException::FATAL) {
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

		if (e.getErrorLevel() == CoreException::FATAL) {
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

		if (e.getErrorLevel() == CoreException::FATAL) {
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

		if (e.getErrorLevel() == CoreException::FATAL) {
			_log_server->error() << "Shutting down server due to error: " + string_t_to_string(e.getErrorMessage());
			exit(0);
		}
	}
}

void listener::init_restmap() {
	try {
		ifstream ini_file("ini/restmap.ini");
		string s;
		string current_factory = "";
		while (std::getline(ini_file, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				current_factory = s;
			}
			else {
				string_t ss(s.begin(), s.end());
				string_t s_current_factory(current_factory.begin(), current_factory.end());
				_path_to_handler[ss] = s_current_factory;
				_log_server->info() << "add mapping from " + s + " to \"" + current_factory + "\"";
			}
		}
	}
	catch (exception &e) {
		throw ServerException("Having some problem reading restmap.ini file!", ServerException::FATAL);
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