#include "UMARestHandler.h"
#include "UMAException.h"
#include "RestUtil.h"
#include "Logger.h"

static Logger accessLogger("Access", "log/UMAC_access.log");
static Logger serverLogger("Server", "log/UMA_server.log");

UMARestHandler::UMARestHandler(const string &handlerName) : _handlerName(handlerName) {

}

const string &UMARestHandler::getHandlerName() const {
	return _handlerName;
}

UMARestHandler::~UMARestHandler() {}