#include "logManager.h"
#include "logging.h"

/*
log manager class
*/

logManager::logManager(){}

logManager::logManager(int sim_level, string path, string filename, string classname){
	_sim_level = sim_level;
	_filename = filename;
	_classname = classname;
	
#if defined(_WIN64)
	_mkdir(path.c_str());
#else 
	mkdir(path.c_str(), 0777);
#endif

	_log = logging(path + "/" + filename, classname);
}

logging &logManager::debug(){
	_log._active = 3 <= _sim_level;
	_log._level = "DEBUG";
	return _log;
}

logging &logManager::info(){
	_log._active = 2 <= _sim_level;
	_log._level = "INFO";
	return _log;
}

logging &logManager::verbose(){
	_log._active = 4 <= _sim_level;
	_log._level = "VERBOSE";
	return _log;
}

logging &logManager::warn() {
	_log._active = 1 <= _sim_level;
	_log._level = "WARN";
	return _log;
}

logging &logManager::error() {
	_log._active = 0 <= _sim_level;
	_log._level = "ERROR";
	return _log;
}