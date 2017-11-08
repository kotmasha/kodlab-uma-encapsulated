#ifndef _UMAUTIL_
#define _UMAUTIL_

#include "Global.h"

#include <cpprest/http_listener.h>
#include <string>
using namespace web::http;
using namespace web;
using namespace utility;
using namespace std;

string_t string_to_string_t(string &s);

string string_t_to_string(string_t &s);

string_t status_code_to_string_t(status_code &s);

std::map<string_t, vector<string_t>> read_restmap();

std::map<string, string> read_log_level();

int string_to_log_level(string &s);

#endif