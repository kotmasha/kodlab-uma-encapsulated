#include "ConfReader.h"
#include "UMAException.h"
#include "PropertyMap.h"

/*
##################ConfReader####################
*/
/*
Read the configuration file of restmap.ini
output: map of string to vector<string>, stanza_name to stanza
*/
PropertyPage *ConfReader::readRestmap() {
	PropertyPage *results = new PropertyPage();
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
				results->add(sCurrentFactory, new PropertyMap());
			}
			else {
				string sPath(s.begin(), s.end());
				results->get(sCurrentFactory)->add(sPath, "");
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
PropertyPage *ConfReader::readConf(const string &confName) {
	PropertyPage *results = new PropertyPage();
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
				results->add(s, new PropertyMap());
				currentComponent = s;
			}
			else {
				string key = s.substr(0, s.find("="));
				string value = s.substr(s.find("=") + 1);
				//nasty way to trim spaces, need to improve it as common lib
				if (key.back() == ' ') key = key.substr(0, key.size() - 1);
				if (value.front() == ' ') value = value.substr(1, value.size());
				results->get(currentComponent)->add(key, value);
			}
		}
		//cout << confName + " read complete" << endl;
		return results;
	}
	catch (exception &e) {
		throw UMAInternalException("Having some problem reading " + confName);
	}
}

/*
##################ConfReader####################
*/