#ifndef _CONFREADER_
#define _CONFREADER_

#include "Global.h"
using namespace std;

class PropertyMap;

namespace ConfReader {
	std::map<string, vector<string>> readRestmap();
	std::map<string, PropertyMap*> readConf(const string &conf_name);
}

#endif
