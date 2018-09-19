#include "LogService.h"
#include "ConfReader.h"
#include "UMAException.h"
#include "UMAutil.h"
#include "PropertyMap.h"

LogService *LogService::_logService = nullptr;

LogService::LogService() {
	try {
		string logFolder = "log";
		SysUtil::UMAMkdir(logFolder);
	}
	catch (exception &ex) {
		cout << "Cannot make a log folder, error=" + string(ex.what()) << endl;
	}
	
	_logLevel = ConfReader::readConf("log.ini");
	cout << "LogService is initiated!" << endl;
}

LogService::~LogService() {}

LogService *LogService::instance() {
	if (!_logService) {
		_logService = new LogService();
	}
	return _logService;
}

string LogService::getLogLevelString(const string &component) {
	string level;
	PropertyMap *ppm = _logLevel->get(component);
	if (!ppm) {
		throw UMAInternalException("Cannot find the component=" + component, false);
	}

	level = ppm->get("level");
	if (StrUtil::isEmpty(level)) {
		throw UMAInternalException("Cannot find the key=level", false);
	}
	return level;
}

/*
input: string level
output: int level
*/
Logger::LOG_LEVEL LogService::stringToLogLevel(const string &s) {
	if (s == "ERROR") return Logger::LOG_ERROR;
	if (s == "WARN") return Logger::LOG_WARN;
	if (s == "INFO") return Logger::LOG_INFO;
	if (s == "DEBUG") return Logger::LOG_DEBUG;
	if (s == "VERBOSE") return Logger::LOG_VERBOSE;
}