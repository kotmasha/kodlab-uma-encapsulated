#ifndef _CONFREADER_
#define _CONFREADER_

#include "Global.h"
#include <cpprest/http_listener.h>
using namespace utility;
using namespace std;

namespace ConfReader {
	std::map<string_t, vector<string_t>> read_restmap();
	std::map<string, std::map<string, string>> read_log_configure();
	std::map<string, std::map<string, string>> read_server_configure();
}

#endif
