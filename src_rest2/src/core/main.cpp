#include "Global.h"
#include "listener.h"
#include "World.h"
#include "LogManager.h"

using namespace std;

#include<sstream>

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

std::map<string, std::map<string, string>> read_log_configure() {
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
		return results;
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
	std::map<string, std::map<string, string>> log_cfg = read_log_configure();

	LogManager *logManager = new LogManager(log_cfg);

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