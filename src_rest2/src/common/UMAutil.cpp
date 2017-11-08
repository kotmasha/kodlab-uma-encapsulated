#include "UMAutil.h"
#include "UMAException.h"

string_t string_to_string_t(string &s) {
	string_t ss(s.begin(), s.end());
	return ss;
}

string string_t_to_string(string_t &s) {
	string ss(s.begin(), s.end());
	return ss;
}

string_t status_code_to_string_t(status_code &s) {
	string ss = to_string(s);
	return string_to_string_t(ss);
}