#include "ConfReader.h"
#include "UMAException.h"

/*
##################ConfReader####################
*/
/*
Read the configuration file of restmap.ini
output: map of string to vector<string>, stanza_name to stanza
*/
std::map<string, vector<string>> ConfReader::readRestmap() {
	std::map<string, vector<string>> results;
	try {
		ifstream iniFile("ini/restmap.ini");
		string s;
		string sCurrentFactory;
		while (std::getline(iniFile, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				sCurrentFactory = string(s.begin(), s.end());
				results[sCurrentFactory] = vector<string>();
			}
			else {
				string sPath(s.begin(), s.end());
				results[sCurrentFactory].push_back(sPath);
			}
		}
		cout << "restmap.ini read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw UMAInternalException("Having some problem reading restmap.ini file!");
	}
}

/*
Reading the log configuration file
Output: map of string to map of string, indicating the log level and etc
*/
std::map<string, std::map<string, string>> ConfReader::readConf(const string &confName) {
	std::map<string, std::map<string, string>> results;
	try {
		ifstream iniFile("ini/" + confName);
		if (!iniFile.good()) {
			cout << "The ini/" + confName + " does not exist!" << endl;
			throw UMAInternalException("The ini/" + confName + " does not exist!");
		}
		string s;
		string currentComponent;
		while (std::getline(iniFile, s)) {
			if (s.front() == '#' || s.length() == 0) continue;
			else if (s.front() == '[' && s.back() == ']') {//get a factory
				s.erase(s.begin());
				s.erase(s.end() - 1);
				results[s] = std::map<string, string>();
				currentComponent = s;
			}
			else {
				string key = s.substr(0, s.find("="));
				string value = s.substr(s.find("=") + 1);
				//nasty way to trim spaces, need to improve it as common lib
				if (key.back() == ' ') key = key.substr(0, key.size() - 1);
				if (value.front() == ' ') value = value.substr(1, value.size());
				results[currentComponent][key] = value;
			}
		}
		cout << confName + " read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw UMAInternalException("Having some problem reading " + confName);
	}
}

/*
##################ConfReader####################
*/