#include "ConfReader.h"
#include "UMAException.h"

/*
Reading the rest map
*/
std::map<string_t, vector<string_t>> ConfReader::read_restmap() {
	std::map<string_t, vector<string_t>> results;
	try {
		ifstream ini_file("ini/restmap.ini");
		string s;
		string_t s_current_factory;
		while (std::getline(ini_file, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				s_current_factory = string_t(s.begin(), s.end());
				results[s_current_factory] = vector<string_t>();
			}
			else {
				string_t s_path(s.begin(), s.end());
				results[s_current_factory].push_back(s_path);
			}
		}
		cout << "restmap read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw ServerException("Having some problem reading restmap.ini file!", ServerException::SERVER_FATAL);
	}
}

/*
Reading the log configuration file
*/
std::map<string, std::map<string, string>> ConfReader::read_log_configure() {
	std::map<string, std::map<string, string>> results;
	try {
		ifstream ini_file("ini/log.ini");
		string s;
		string current_component;
		while (std::getline(ini_file, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				results[s] = std::map<string, string>();
				current_component = s;
			}
			else {
				string key = s.substr(0, s.find("="));
				string value = s.substr(s.find("=") + 1);
				results[current_component][key] = value;
			}
		}
		cout << "log configuration read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw ServerException("Having some problem reading restmap.ini file!", ServerException::SERVER_FATAL);
	}
}

/*
Reading the server configuration
*/
std::map<string, std::map<string, string>> ConfReader::read_server_configure() {
	std::map<string, std::map<string, string>> results;
	try {
		ifstream ini_file("ini/server.ini");
		string s;
		string current_obj;
		while (std::getline(ini_file, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				results[s] = std::map<string, string>();
				current_obj = s;
			}
			else {
				string key = s.substr(0, s.find("="));
				string value = s.substr(s.find("=") + 1);
				results[current_obj][key] = value;
			}
		}
		cout << "server configuration read complete" << endl;
		return results;
	}
	catch (exception &e) {
		//throw ServerException("Having some problem reading restmap.ini file!", ServerException::SERVER_FATAL);
	}
}