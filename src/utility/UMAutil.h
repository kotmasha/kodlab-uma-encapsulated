#ifndef _UMAUTIL_
#define _UMAUTIL_

//#define CONSOLETEST_API __declspec(dllexport)

/*
This is the UMA utility class, please add all utility functions inside
*/

#include "Global.h"

namespace SysUtil {
	//mkdir under different platform
	string UMA_mkdir(std::string path);
}

namespace StrUtil {
	//conversion of string to log_level
	int string_to_log_level(const string &s);
	vector<std::pair<string, string>> string2d_to_string1d_pair(const vector<vector<string>> &pairs);
}

namespace SignalUtil {
	const vector<int> bool_signal_to_int_idx(const vector<bool> &list);
	const vector<bool> measurable_signal_to_sensor_signal(const vector<bool> &signal);
}

#endif