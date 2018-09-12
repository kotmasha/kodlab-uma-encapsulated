#ifndef _LOGSERVICE_
#define _LOGSERVICE_

#include "Global.h"

class LogService {
private:
	static LogService *_logService;
	std::map < string, std::map<string, string>> _logLevel;

public:
	LogService();
	~LogService();
	static LogService *instance();
	string getLogLevelString(const string &component);
	int stringToLogLevel(const string &s);
};

#endif