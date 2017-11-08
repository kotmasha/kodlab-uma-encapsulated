#include "Global.h"
#include "listener.h"
#include "World.h"
#include "logManager.h"

using namespace std;

#include<sstream>

std::map<string, int> log_level;
extern logManager *sim_log;

int string_to_log_level(string &s) {
	if (s == "ERROR") return 0;
	if (s == "WARN") return 1;
	if (s == "INFO") return 2;
	if (s == "DEBUG") return 3;
	if (s == "VERBOSE") return 4;
}

string_t convert_string(string s) {
	string_t ss(s.begin(), s.end());
	return ss;
}

void read_ini(string_t &port, string_t &host) {
	try {
		ifstream ini_file("ini.txt");
		string s;
		while (std::getline(ini_file, s)){
			if (s.front() == '#') continue;
			if (s.find("port") != std::string::npos) {
				std::getline(ini_file, s);
				port = convert_string(s);
			}
			else if (s.find("host") != std::string::npos) {
				std::getline(ini_file, s);
				host = convert_string(s);
			}
		}
	}
	catch (exception &e) {
		cout << "Cannot find the ini.txt file, will use default settings" << endl;
	}
}


std::map<string_t, vector<string_t>> read_restmap() {
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
		return results;
	}
	catch (exception &e) {
		//throw ServerException("Having some problem reading restmap.ini file!", ServerException::SERVER_FATAL);
	}
}

void read_log_level() {
	try {
		ifstream ini_file("ini/log.ini");
		string s;
		while (std::getline(ini_file, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
														   //for now escape, it is only [log_level] for now
			}
			else {
				string obj = s.substr(0, s.find("="));
				string level = s.substr(s.find("=") + 1);
				log_level[obj] = string_to_log_level(level);
			}
		}
		//hack if for now
		if (log_level.find("Server") == log_level.end()) log_level["Server"] = 2;
		if (log_level.find("World") == log_level.end()) log_level["World"] = 2;
		if (log_level.find("Agent") == log_level.end()) log_level["Agent"] = 2;
		if (log_level.find("Snapshot") == log_level.end()) log_level["Snapshot"] = 2;
		if (log_level.find("DataManager") == log_level.end()) log_level["DataManager"] = 2;
		if (log_level.find("Simulation") == log_level.end()) log_level["Simulation"] = 2;
	}
	catch (exception &e) {
		//throw ServerException("Having some problem reading restmap.ini file!", ServerException::SERVER_FATAL);
	}
}

int main() {
	string_t port = U("8000"), host=U("localhost");
	read_ini(port, host);
	uri url = uri(U("http://") + host + U(":") + port);

	std::map<string_t, vector<string_t>> rest_map = read_restmap();
	read_log_level();

	sim_log = new logManager(log_level["Simulation"], "log", "simulation.txt", "Simulation");

	listener listener(url, rest_map);
	try
	{
		listener.m_listener.open().wait();
		while (true);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}