#include "UMARestListener.h"
#include "UMAException.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"
#include "RestUtil.h"
#include "Logger.h"

using namespace std;

static Logger accessLogger("Access", "log/UMAC_access.log");
static Logger serverLogger("Server", "log/UMA_server.log");

UMARestListener::UMARestListener(const string &url) : _listener(uri(RestUtil::string_to_string_t(url))) {
	serverLogger.info("Listening on the url " + url);
	_listener.support(methods::GET, std::bind(&UMARestListener::handle_get, this, std::placeholders::_1));
	serverLogger.info("Init Get request success");

	_listener.support(methods::POST, std::bind(&UMARestListener::handle_post, this, std::placeholders::_1));
	serverLogger.info("Init Post request success");

	_listener.support(methods::PUT, std::bind(&UMARestListener::handle_put, this, std::placeholders::_1));
	serverLogger.info("Init Put request success");

	_listener.support(methods::DEL, std::bind(&UMARestListener::handle_delete, this, std::placeholders::_1));
	serverLogger.info("Init Delete request success");
}

UMARestListener::~UMARestListener() {}

void UMARestListener::register_handler(UMARestHandler *handler) {
	if (_registered_handlers.find(handler->getHandlerName()) != _registered_handlers.end()) {
		serverLogger.error("Handler with the name " + handler->getHandlerName() + " already exist!");
		exit(0);
	}
	_registered_handlers[handler->getHandlerName()] = handler;
	serverLogger.info("Registered handler with the name " + handler->getHandlerName());
}

void UMARestListener::add_path_to_handler(const string &path, const string &handler_name) {
	if (_registered_handlers.find(handler_name) == _registered_handlers.end()) {
		serverLogger.warn("Cannot find the handler by the handler name " + handler_name);
		return;
	}
	_path_to_handler[path] = _registered_handlers[handler_name];
	serverLogger.info("add mapping from " + path + " to \"" + handler_name + "\"");
}

void UMARestListener::listen() {
	_listener.open().then([](pplx::task<void> t) {});
}

void UMARestListener::handle(http_request &request, string request_type) {
	UMARestRequest uma_request = UMARestRequest(request);
	string url_path = uma_request.get_request_url();
	UMARestHandler *handler = find_handler(url_path);

	// if no handlers can be found
	if (handler == NULL) {
		accessLogger.error(request_type + " " + uma_request.get_absolute_url() + " 400");
		uma_request.set_status_code(status_codes::BadRequest);
		uma_request.set_message("cannot find coresponding handler!");

		uma_request.reply();
		return;
	}

	try {
		if (request_type == "POST") {
			handler->handleCreate(uma_request);
			uma_request.set_status_code(status_codes::Created);
			accessLogger.info(request_type + " " + uma_request.get_absolute_url() + " " + RestUtil::status_code_to_string(status_codes::Created));
		}
		else if (request_type == "PUT") {
			handler->handleUpdate(uma_request);
			uma_request.set_status_code(status_codes::OK);
			accessLogger.info(request_type + " " + uma_request.get_absolute_url() + " " + RestUtil::status_code_to_string(status_codes::OK));
		}
		else if (request_type == "GET") {
			handler->handleRead(uma_request);
			uma_request.set_status_code(status_codes::OK);
			accessLogger.info(request_type + " " + uma_request.get_absolute_url() + " " + RestUtil::status_code_to_string(status_codes::OK));
		}
		else {
			handler->handleDelete(uma_request);
			uma_request.set_status_code(status_codes::OK);
			accessLogger.info(request_type + " " + uma_request.get_absolute_url() + " " + RestUtil::status_code_to_string(status_codes::OK));
		}
		uma_request.reply();
	}
	catch (UMAException &e) {
		const status_code code = RestUtil::UMAExceptionToStatusCode(&e);
		uma_request.set_status_code(code);
		uma_request.set_message(e.getErrorMessage());
		if (!e.isErrorLogged()) {
			serverLogger.error(e.getErrorMessage());
		}
		accessLogger.error(request_type + " " + uma_request.get_absolute_url() + " " + RestUtil::status_code_to_string(code));

		uma_request.reply();

		if (e.isFatal()) {
			serverLogger.error("Shutting down server due to error: " + e.getErrorMessage());
			exit(0);
		}
	}
	catch (exception &e) {
		status_code code = status_codes::InternalError;
		serverLogger.error(e.what());
		accessLogger.error("POST " + uma_request.get_absolute_url() + " " + RestUtil::status_code_to_string(code));
		uma_request.set_status_code(code);
		uma_request.set_message(string(e.what()));

		uma_request.reply();

		serverLogger.error("Shutting down server due to RUNTIME error: " + string(e.what()));
		exit(0);
	}
}

void UMARestListener::handle_get(http_request request) {
	handle(request, "GET");
}

void UMARestListener::handle_put(http_request request) {
	handle(request, "PUT");
}

void UMARestListener::handle_post(http_request request) {
	handle(request, "POST");
}

void UMARestListener::handle_delete(http_request request) {
	handle(request, "DELETE");
}

UMARestHandler *UMARestListener::find_handler(const string &path) {
	if (_path_to_handler.find(path) == _path_to_handler.end()) {
		return NULL;
	}
	return _path_to_handler[path];
}
