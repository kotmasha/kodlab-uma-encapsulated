#ifndef _LOGGER_
#define _LOGGER_

#include "Global.h"
using namespace std;

class Logger {
private:
	ofstream *_output;
	string const _component;
	int _level;

public:
	enum { LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_VERBOSE };

public:
	Logger(const string component, const string output);
	void verbose(string message, const string &ancestors = "") const;
	void debug(string message, const string &ancestors = "") const;
	void info(string message, const string &ancestors = "") const;
	void warn(string message, const string &ancestors = "") const;
	void error(string message, const string &ancestors = "") const;
	string getTime() const;
	void setLogLevel(int level);
};

#endif
