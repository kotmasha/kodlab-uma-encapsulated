#include "ConfReader.h"
#include "UMAException.h"

/*
##################ConfReader####################
*/
/*
Read the configuration file of restmap.ini
output: map of string to vector<string>, stanza_name to stanza
*/
std::map<string, vector<string>> ConfReader::read_restmap() {
	std::map<string, vector<string>> results;
	try {
		ifstream ini_file("ini/restmap.ini");
		string s;
		string s_current_factory;
		while (std::getline(ini_file, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				s_current_factory = string(s.begin(), s.end());
				results[s_current_factory] = vector<string>();
			}
			else {
				string s_path(s.begin(), s.end());
				results[s_current_factory].push_back(s_path);
			}
		}
		cout << "restmap.ini read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw UMAException("Having some problem reading restmap.ini file!", UMAException::FATAL, UMAException::CONF_ERROR);
	}
}

/*
Reading the log configuration file
Output: map of string to map of string, indicating the log level and etc
*/
std::map<string, std::map<string, string>> ConfReader::read_conf(const string &conf_name) {
	std::map<string, std::map<string, string>> results;
	try {
		ifstream ini_file("ini/" + conf_name);
		if (!ini_file.good()) {
			cout << "The ini/" + conf_name + " does not exist!" << endl;
			throw UMAException("The ini/" + conf_name + " does not exist!", UMAException::FATAL, UMAException::CONF_ERROR);
		}
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
		cout << conf_name + " read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw UMAException("Having some problem reading " + conf_name, UMAException::FATAL, UMAException::CONF_ERROR);
	}
}

/*
##################ConfReader####################
*/