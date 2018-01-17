#include "UMARestHandler.h"
#include "UMAException.h"
#include "RestUtil.h"
#include "Logger.h"

static Logger accessLogger("Access", "log/UMAC_access.log");
static Logger serverLogger("Server", "log/UMA_server.log");

UMARestHandler::UMARestHandler(const string &handler_name) : _handler_name(handler_name) {

}

const string &UMARestHandler::getHandlerName() const {
	return _handler_name;
}

UMARestHandler::~UMARestHandler() {}