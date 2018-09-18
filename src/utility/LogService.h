#ifndef _LOGSERVICE_
#define _LOGSERVICE_

#include "Global.h"
#include "Logger.h"

class PropertyPage;

class LogService {
private:
	static LogService *_logService;
	PropertyPage *_logLevel;

public:
	LogService();
	~LogService();
	static LogService *instance();
	string getLogLevelString(const string &component);
	Logger::LOG_LEVEL stringToLogLevel(const string &s);
};

#endif