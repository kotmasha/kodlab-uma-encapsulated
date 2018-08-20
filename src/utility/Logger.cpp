#include "Logger.h"
#include "UMAutil.h"
#include "ConfReader.h"
#include "UMAException.h"
#include <ctime>
#include <time.h>
#include <chrono>

Logger::Logger(const string component, const string output) : _component(component) {
	//level is changable in runtime
	static std::map < string, std::map<string, string>> _logLevel = ConfReader::readConf("log.ini");
	SysUtil::UMAMkdir(string("log"));

	try {
		_level = StrUtil::stringToLogLevel(_logLevel[component]["level"]);
	}
	catch (UMAException &e) {
		cout << "Cannot find the component " + component << endl;
		exit(0);
        std::getchar();
	}
	_output = new ofstream(output, std::ios_base::app);
}

string Logger::getTime() const{
	auto current_time = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(current_time);
	char tStr[26];
#if defined(_WIN64)
	ctime_s(tStr, sizeof tStr, &t);
	return string(tStr).substr(0, 24);
#else
	return string(ctime(&t)).substr(0, 24);
#endif
}

void Logger::verbose(string message) const{
	if (_level < LOG_VERBOSE) return;
	message = getTime() + " VERBOSE " + _component + " - " + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::debug(string message) const{
	if (_level < LOG_DEBUG) return;
	message = getTime() + " DEBUG " + _component + " - " + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::info(string message) const{
	if (_level < LOG_INFO) return;
	message = getTime() + " INFO " + _component + " - " + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::warn(string message) const{
	if (_level < LOG_WARN) return;
	message = getTime() + " WARN " + _component + " - " + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::error(string message) const{
	if (_level < LOG_ERROR) return;
	message = getTime() + " ERROR " + _component + " - " + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::setLogLevel(int level) {
	_level = level;
}
