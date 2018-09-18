#ifndef _LOGGER_
#define _LOGGER_

#include "Global.h"
using namespace std;

class Logger {
public:
	enum LOG_LEVEL { LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_VERBOSE };

private:
	ofstream *_output;
	string const _component;
	LOG_LEVEL _level;

public:
	Logger(const string component, const string output);
	void verbose(string message, const string &ancestors = "") const;
	void debug(string message, const string &ancestors = "") const;
	void info(string message, const string &ancestors = "") const;
	void warn(string message, const string &ancestors = "") const;
	void error(string message, const string &ancestors = "") const;
	string getTime() const;
	void setLogLevel(LOG_LEVEL level);
	Logger::LOG_LEVEL getLogLevel();
};

#endif
