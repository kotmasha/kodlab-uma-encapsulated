#ifndef _CONFREADER_
#define _CONFREADER_

#include "Global.h"
#include "PropertyPage.h"
using namespace std;

namespace ConfReader {
	std::map<string, vector<string>> readRestmap();
	PropertyPage *readConf(const string &conf_name);
}

#endif
