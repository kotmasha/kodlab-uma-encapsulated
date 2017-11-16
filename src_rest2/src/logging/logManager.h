#ifndef _LOGMANAGER_
#define _LOGMANAGER_

#include "Global.h"
using namespace std;

class LogManager{
private:
	std::map<string, std::map<string, string>> _log_cfg;
public:
	enum{ERROR, WARN, INFO, DEBUG, VERBOSE};

public:
	LogManager();
	void init_log_dirs();
	void init_logger();
};

#endif