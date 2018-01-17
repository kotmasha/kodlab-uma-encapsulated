#include "Logger.h"
#include "UMAutil.h"
#include "ConfReader.h"
#include "UMAException.h"
#include <ctime>
#include <time.h>
#include <chrono>

Logger::Logger(const string component, const string output) : _component(component) {
	//level is changable in runtime
	static std::map < string, std::map<string, string>> _log_level = ConfReader::read_conf("log.ini");
	SysUtil::UMA_mkdir("log");

	try {
		_level = StrUtil::string_to_log_level(_log_level[component]["level"]);
	}
	catch (UMAException &e) {
		cout << "Cannot find the component " + component << endl;
		exit(0);
		system("pause");
	}
	_output = new ofstream(output, std::ios_base::app);
}

string Logger::getTime() const{
	auto current_time = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(current_time);
	char t_str[26];
#if defined(_WIN64)
	ctime_s(t_str, sizeof t_str, &t);
	return string(t_str).substr(0, 24);
#else
	return string(ctime(&t)).substr(0, 24);
#endif
}

void Logger::verbose(string message, string dependency) const{
	if (_level < LOG_VERBOSE) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " VERBOSE " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::debug(string message, string dependency) const{
	if (_level < LOG_DEBUG) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " DEBUG " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::info(string message, string dependency) const{
	if (_level < LOG_INFO) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " INFO " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::warn(string message, string dependency) const{
	if (_level < LOG_WARN) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " WARN " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::error(string message, string dependency) const{
	if (_level < LOG_ERROR) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " ERROR " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::setLogLevel(int level) {
	_level = level;
}