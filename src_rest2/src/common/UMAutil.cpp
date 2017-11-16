#include "UMAutil.h"
#include "UMAException.h"

string_t string_to_string_t(string s) {
	string_t ss(s.begin(), s.end());
	return ss;
}

string string_t_to_string(string_t s) {
	string ss(s.begin(), s.end());
	return ss;
}

string_t status_code_to_string_t(status_code s) {
	string ss = to_string(s);
	return string_to_string_t(ss);
}

void UMA_mkdir(string path) {
#if defined(_WIN64)
	_mkdir(path.c_str());
#else 
	mkdir(path.c_str(), 0777);
#endif
}


int string_to_log_level(string &s) {
	if (s == "ERROR") return 0;
	if (s == "WARN") return 1;
	if (s == "INFO") return 2;
	if (s == "DEBUG") return 3;
	if (s == "VERBOSE") return 4;
}