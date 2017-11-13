#include "Logger.h"
#include "LogManager.h"
#include <ctime>
#include <time.h>
#include <chrono>

extern LogManager *logManager;

Logger::Logger(){}

Logger::Logger(string component, string filename, int level) {
	_component = component;
	_filename = "log/" + filename;
	_level = level;
	_output = new ofstream(_filename);
}

string Logger::getTime() {
	auto current_time = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(current_time);
	char t_str[26];
	ctime_s(t_str, sizeof t_str, &t);
	return string(t_str).substr(0, 24);
}

void Logger::verbose(string message, string dependency) {
	if (_level < LogManager::VERBOSE) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " VERBOSE " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::debug(string message, string dependency) {
	if (_level < LogManager::DEBUG) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " DEBUG " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::info(string message, string dependency) {
	if (_level < LogManager::INFO) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " INFO " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::warn(string message, string dependency) {
	if (_level < LogManager::WARN) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " WARN " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}

void Logger::error(string message, string dependency) {
	if (_level < LogManager::ERROR) return;
	string dep = "[" + dependency + "]";
	if (dependency == "") dep = "";
	message = getTime() + " ERROR " + _component + " - " + dep + message + "\n";
	_output->write(message.c_str(), message.size() * sizeof(char));
	_output->flush();
}