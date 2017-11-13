#ifndef _LOGMANAGER_
#define _LOGMANAGER_

#include "Global.h"
using namespace std;

class LogManager{
private:
	std::map<string, std::map<string, string>> _log_cfg;
public:
	enum{ERROR, WARN, INFO, DEBUG, VERBOSE};

private:
	void init_log_files();
	void init_logger();
public:
	LogManager();
	LogManager(std::map<string, std::map<string, string>> &log_cfg);
};

#endif