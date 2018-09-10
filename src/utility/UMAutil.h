#ifndef _UMAUTIL_
#define _UMAUTIL_

//#define CONSOLETEST_API __declspec(dllexport)

/*
This is the UMA utility class, please add all utility functions inside
*/

#include "Global.h"

namespace SysUtil {
	//mkdir under different platform
	string UMAMkdir(std::string &path);
	bool UMAFileExist(std::string &path);
	bool UMAFolderExist(std::string &path);
}

namespace StrUtil {
	//conversion of string to log_level
	vector<std::pair<string, string>> string2dToString1dPair(const vector<vector<string>> &pairs);
}

namespace SignalUtil {
	const vector<int> boolSignalToIntIdx(const vector<bool> &list);
	const vector<bool> intIdxToBoolSignal(const vector<int> &idx, int size);
	const vector<bool> attrSensorToSensorSignal(const vector<bool> &signal);
	const vector<bool> trimSignal(const vector<bool> &signal);
}

namespace ArrayUtil {
	int findIdxInSortedArray(const vector<int> &input, int target);
}

#endif