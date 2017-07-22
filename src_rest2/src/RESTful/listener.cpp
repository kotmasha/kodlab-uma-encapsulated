#include "listener.h"
#include "World.h"
#include "DataHandler.h"
#include "ObjectHandler.h"
#include "DataValidationHandler.h"
#include "SimulationHandler.h"
#include "AdminHandler.h"
#include "logManager.h"
#include <cpprest/json.h>

listener::listener(const http::uri& url) : m_listener(http_listener(url)){
	//init the UMAC_access and UMAC log
	_log_path = "log";
	_log_access = new logManager(logging::VERBOSE, _log_path, "UMAC_access.txt", typeid(*this).name());
	_log_server = new logManager(logging::VERBOSE, _log_path, "UMA_server.txt", typeid(*this).name());
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

	_data_handler = new DataHandler(_log_access);
	_log_server->info() << "A data handler is created";

	_object_handler = new ObjectHandler(_log_access);
	_log_server->info() << "A object handler is created";

	_data_validation_handler = new DataValidationHandler(_log_access);
	_log_server->info() << "A data validation handler is created";

	_simulation_handler = new SimulationHandler(_log_access);
	_log_server->info() << "A simulation handler is created";
}

void listener::handle_get(http_request request){
	vector<string_t> paths = uri::split_path(request.request_uri().path());

	AdminHandler *handler = find_handler(paths);
	if (handler == NULL) {
		_log_access->error() << request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	handler->REQUEST_MODE = handler->GET;
	handler->handle_read(_world, paths, request);
}

void listener::handle_put(http_request request){
	vector<string_t> paths = uri::split_path(request.request_uri().path());

	AdminHandler *handler = find_handler(paths);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	handler->REQUEST_MODE = handler->PUT;
	handler->handle_update(_world, paths, request);
}

void listener::handle_post(http_request request){
	vector<string_t> paths = uri::split_path(request.request_uri().path());

	AdminHandler *handler = find_handler(paths);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << request.absolute_uri().to_string() + U(" 400");
		json::value message;
		message[U("message")] = json::value::string(U("cannot find coresponding handler!"));
		request.reply(status_codes::BadRequest, message);
		return;
	}
	handler->REQUEST_MODE = handler->POST;
	handler->handle_create(_world, paths, request);
}

void listener::handle_delete(http_request request){
	request.reply(status_codes::NotFound);
}

AdminHandler *listener::find_handler(vector<string_t> &paths) {
	if (paths[0] != U("UMA") || paths.size() < 2) {
		return NULL;
	}
	paths.erase(paths.begin());
	if (paths[0] == U("data")) {
		paths.erase(paths.begin());
		return _data_handler;
	}
	else if (paths[0] == U("object")) {
		paths.erase(paths.begin());
		return _object_handler;
	}
	else if (paths[0] == U("simulation")) {
		paths.erase(paths.begin());
		return _simulation_handler;
	}
	else if (paths[0] == U("validation")) {
		paths.erase(paths.begin());
		return  _data_validation_handler;
	}
	return NULL;
}