#include "listener.h"
#include "World.h"
#include "DataHandler.h"
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
	m_listener.support(methods::GET, std::tr1::bind(&listener::handle_get, this, std::tr1::placeholders::_1));
	_log_server->info() << "Init Get request success";

	m_listener.support(methods::PUT, std::tr1::bind(&listener::handle_put, this, std::tr1::placeholders::_1));
	_log_server->info() << "Init Put request success";

	m_listener.support(methods::POST, std::tr1::bind(&listener::handle_post, this, std::tr1::placeholders::_1));
	_log_server->info() << "Init Post request success";

	m_listener.support(methods::DEL, std::tr1::bind(&listener::handle_delete, this, std::tr1::placeholders::_1));
	_log_server->info() << "Init Delete request success";
	//create data handler

	_data_handler = new DataHandler(_log_access);
	_log_server->info() << "A data handler is created";

	_data_validation_handler = new DataValidationHandler(_log_access);
	_log_server->info() << "A data validation handler is created";

	_simulation_handler = new SimulationHandler(_log_access);
	_log_server->info() << "A simulation handler is created";
}

void listener::handle_get(http_request message){
	std::cout << "a request is coming in" << std::endl;
	std::wcout << message.absolute_uri().to_string() << std::endl;
	std::wcout << message.relative_uri().to_string() << std::endl;
	message.reply(status_codes::BadRequest);
}

void listener::handle_put(http_request message){
	message.reply(status_codes::NotFound);
	const web::uri &s = message.relative_uri();
}

void listener::handle_post(http_request request){
	vector<string_t> &paths = uri::split_path(request.relative_uri().to_string());

	AdminHandler *handler = find_handler(paths);
	// if no handlers can be found
	if (handler == NULL) {
		_log_access->error() << request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"cannot find coresponding handler!"));
		return;
	}
	handler->handle_create(_world, paths, request);
}

void listener::handle_delete(http_request message){
	message.reply(status_codes::NotFound);
}

AdminHandler *listener::find_handler(vector<string_t> &paths) {
	if (paths[0] != L"UMA" || paths.size() < 2) {
		return NULL;
	}
	paths.erase(paths.begin());
	if (paths[0] == L"data") {
		paths.erase(paths.begin());
		return _data_handler;
	}
	else if (paths[0] == L"simulation") {
		paths.erase(paths.begin());
		return _simulation_handler;
	}
	else if (paths[0] == L"validation") {
		paths.erase(paths.begin());
		return  _data_validation_handler;
	}
	return NULL;
}