#ifndef _LOGSERVICE_
#define _LOGSERVICE_

#include "Global.h"

class PropertyMap;

class LogService {
private:
	static LogService *_logService;
	std::map < string, PropertyMap*> _logLevel;

public:
	LogService();
	~LogService();
	static LogService *instance();
	string getLogLevelString(const string &component);
	int stringToLogLevel(const string &s);
};

#endif