#ifndef _CONFREADER_
#define _CONFREADER_

#include "Global.h"
using namespace std;

namespace ConfReader {
	std::map<string, vector<string>> readRestmap();
	std::map<string, std::map<string, string>> readConf(const string &conf_name);
}

#endif
