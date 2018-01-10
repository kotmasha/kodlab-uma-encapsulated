#ifndef _CONFREADER_
#define _CONFREADER_

#include "Global.h"
using namespace std;

namespace ConfReader {
	std::map<string, vector<string>> read_restmap();
	std::map<string, std::map<string, string>> read_conf(const string &conf_name);
}

#endif
