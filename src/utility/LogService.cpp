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
	try {
		level = _logLevel[component]->get("level");
		return level;
	}
	catch (UMAInternalException &e) {
		cout << "Cannot find the component " + component << endl;
		exit(0);
		std::getchar();
	}
}

/*
input: string level
output: int level
*/
int LogService::stringToLogLevel(const string &s) {
	if (s == "ERROR") return 0;
	if (s == "WARN") return 1;
	if (s == "INFO") return 2;
	if (s == "DEBUG") return 3;
	if (s == "VERBOSE") return 4;
}