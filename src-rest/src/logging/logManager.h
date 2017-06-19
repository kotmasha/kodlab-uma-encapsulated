#ifndef _LOGMANAGER_
#define _LOGMANAGER_

#include "Global.h"
#include "logging.h"
using namespace std;

class logManager{
private:
	string _filename;
	string _classname;
	logging _log;
	int _sim_level;
public:

public:
	logManager();
	logManager(int sim_level, string path, string filename, string classname);
	logging &debug();
	logging &verbose();
	logging &info();
	logging &warn();
	logging &error();
};

#endif