#include "logManager.h"
#include "logging.h"

logManager::logManager(){}

logManager::logManager(int sim_level, string path, string filename, string classname){
	_sim_level = sim_level;
	_filename = filename;
	_classname = classname;

	_mkdir(path.c_str());
	_log = logging(path + "/" + filename, classname);
}

logging &logManager::debug(){
	_log._active = logging::DEBUG <= _sim_level;
	_log._level = "DEBUG";
	return _log;
}

logging &logManager::info(){
	_log._active = logging::INFO <= _sim_level;
	_log._level = "INFO";
	return _log;
}

logging &logManager::verbose(){
	_log._active = logging::VERBOSE <= _sim_level;
	_log._level = "VERBOSE";
	return _log;
}