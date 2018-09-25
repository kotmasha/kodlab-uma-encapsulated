#ifndef _CONFREADER_
#define _CONFREADER_

#include "Global.h"
#include "PropertyPage.h"
using namespace std;

namespace ConfReader {
	PropertyPage *readRestmap();
	PropertyPage *readConf(const string &conf_name);
}

#endif
