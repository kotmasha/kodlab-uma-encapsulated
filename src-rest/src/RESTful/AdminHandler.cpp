#include "AdminHandler.h"
#include "logManager.h"

AdminHandler::AdminHandler() {
	NAME = L"name";
	UUID = L"uuid";
	UMA_AGENT = L"agent";
	UMA_SNAPSHOT = L"snapshot";
	UMA_SENSOR = L"sensor";
	UMA_AGENT_ID = L"agent_id";
	UMA_SNAPSHOT_ID = L"snapshot_id";
	UMA_SENSOR_ID = L"sensor_id";
}

AdminHandler::AdminHandler(logManager *log_access):AdminHandler() {
	_log_access = log_access;
}

bool AdminHandler::check_field(json::value &data, string_t &s, http_request &request) {
	if (!data.has_field(s)) {
		_log_access->error()<< request.absolute_uri().to_string() + L" 400";
		request.reply(status_codes::BadRequest, json::value::string(L"Field \'" + s + L"\' must be specified"));
		return false;
	}
	return true;
}

void AdminHandler::parsing_error(http_request &request) {
	_log_access->error() << request.absolute_uri().to_string() + L" 400";
	request.reply(status_codes::BadRequest, json::value::string(L"error parsing the input"));
}

AdminHandler::~AdminHandler() {}