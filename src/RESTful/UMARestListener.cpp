#include "UMARestListener.h"
#include "UMAException.h"
#include "UMARestHandler.h"
#include "UMARestRequest.h"
#include "RestUtil.h"
#include "Logger.h"

using namespace std;

static Logger accessLogger("Access", "log/UMAC_access.log");
static Logger serverLogger("Server", "log/UMA_server.log");

UMARestListener::UMARestListener(const string &url) : _listener(uri(RestUtil::string2string_t(url))) {
	serverLogger.info("Listening on the url " + url);
	_listener.support(methods::GET, std::bind(&UMARestListener::handleGet, this, std::placeholders::_1));
	serverLogger.info("Init Get request success");

	_listener.support(methods::POST, std::bind(&UMARestListener::handlePost, this, std::placeholders::_1));
	serverLogger.info("Init Post request success");

	_listener.support(methods::PUT, std::bind(&UMARestListener::handlePut, this, std::placeholders::_1));
	serverLogger.info("Init Put request success");

	_listener.support(methods::DEL, std::bind(&UMARestListener::handleDelete, this, std::placeholders::_1));
	serverLogger.info("Init Delete request success");
}

UMARestListener::~UMARestListener() {}

void UMARestListener::registerHandler(UMARestHandler *handler) {
	if (_registeredHandlers.find(handler->getHandlerName()) != _registeredHandlers.end()) {
		serverLogger.error("Handler with the name " + handler->getHandlerName() + " already exist!");
		exit(0);
	}
	_registeredHandlers[handler->getHandlerName()] = handler;
	serverLogger.info("Registered handler with the name " + handler->getHandlerName());
}

void UMARestListener::addPathToHandler(const string &path, const string &handlerName) {
	if (_registeredHandlers.find(handlerName) == _registeredHandlers.end()) {
		serverLogger.warn("Cannot find the handler by the handler name " + handlerName);
		return;
	}
	_pathToHandler[path] = _registeredHandlers[handlerName];
	serverLogger.info("add mapping from " + path + " to \"" + handlerName + "\"");
}

void UMARestListener::listen() {
	_listener.open().then([](pplx::task<void> t) {});
}

void UMARestListener::handle(http_request &request, string requestType) {
	UMARestRequest umaRequest = UMARestRequest(request);
	string urlPath = umaRequest.getRequestUrl();
	UMARestHandler *handler = findHandler(urlPath);

	// if no handlers can be found
	if (handler == NULL) {
		accessLogger.error(requestType + " " + umaRequest.getAbsoluteUrl() + " 400");
		umaRequest.setStatusCode(status_codes::BadRequest);
		umaRequest.setMessage("cannot find coresponding handler!");

		umaRequest.reply();
		return;
	}

	try {
		if (requestType == "POST") {
			handler->handleCreate(umaRequest);
			umaRequest.setStatusCode(status_codes::Created);
			accessLogger.info(requestType + " " + umaRequest.getAbsoluteUrl() + " " + RestUtil::status_code2string(status_codes::Created));
		}
		else if (requestType == "PUT") {
			handler->handleUpdate(umaRequest);
			umaRequest.setStatusCode(status_codes::OK);
			accessLogger.info(requestType + " " + umaRequest.getAbsoluteUrl() + " " + RestUtil::status_code2string(status_codes::OK));
		}
		else if (requestType == "GET") {
			handler->handleRead(umaRequest);
			umaRequest.setStatusCode(status_codes::OK);
			accessLogger.info(requestType + " " + umaRequest.getAbsoluteUrl() + " " + RestUtil::status_code2string(status_codes::OK));
		}
		else {
			handler->handleDelete(umaRequest);
			umaRequest.setStatusCode(status_codes::OK);
			accessLogger.info(requestType + " " + umaRequest.getAbsoluteUrl() + " " + RestUtil::status_code2string(status_codes::OK));
		}
		umaRequest.reply();
	}
	catch (UMAException &e) {
		const status_code code = RestUtil::UMAExceptionToStatusCode(&e);
		umaRequest.setStatusCode(code);
		umaRequest.setMessage(e.getErrorMessage());
		if (!e.isErrorLogged()) {
			serverLogger.error(e.getErrorMessage());
		}
		accessLogger.error(requestType + " " + umaRequest.getAbsoluteUrl() + " " + RestUtil::status_code2string(code));

		umaRequest.reply();

		if (e.isFatal()) {
			serverLogger.error("Shutting down server due to error: " + e.getErrorMessage());
			exit(0);
		}
	}
	catch (exception &e) {
		status_code code = status_codes::InternalError;
		serverLogger.error(e.what());
		accessLogger.error("POST " + umaRequest.getAbsoluteUrl() + " " + RestUtil::status_code2string(code));
		umaRequest.setStatusCode(code);
		umaRequest.setMessage(string(e.what()));

		umaRequest.reply();

		serverLogger.error("Shutting down server due to RUNTIME error: " + string(e.what()));
		exit(0);
	}
}

void UMARestListener::handleGet(http_request request) {
	handle(request, "GET");
}

void UMARestListener::handlePut(http_request request) {
	handle(request, "PUT");
}

void UMARestListener::handlePost(http_request request) {
	handle(request, "POST");
}

void UMARestListener::handleDelete(http_request request) {
	handle(request, "DELETE");
}

UMARestHandler *UMARestListener::findHandler(const string &path) {
	if (_pathToHandler.find(path) == _pathToHandler.end()) {
		return NULL;
	}
	return _pathToHandler[path];
}
