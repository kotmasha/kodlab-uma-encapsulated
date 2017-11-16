#ifndef _LOGGER_
#define _LOGGER_

#include "Global.h"
using namespace std;

class Logger {
private:
	ofstream *_output;
	string _component;
	string _filename;
	int _level;

public:
	Logger();
	Logger(string component, string filename, int log_level);
	void verbose(string message, string dependency="");
	void debug(string message, string dependency="");
	void info(string message, string dependency="");
	void warn(string message, string dependency="");
	void error(string message, string dependency="");
	string getTime();
};

#endif